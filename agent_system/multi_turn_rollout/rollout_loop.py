# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import uuid
from contextlib import contextmanager, suppress
from typing import Dict, List

import numpy as np
import torch
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.utils import (
    filter_group_data,
    process_image,
    to_list_of_dict,
    torch_to_numpy,
)
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def _record_session_cleanup_error(primary_error, cleanup_error):
    with suppress(BaseException):
        primary_error.rollout_session_cleanup_error = cleanup_error

    with suppress(BaseException):
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(
                f"rollout session cleanup also failed: "
                f"{cleanup_error!r}"
            )

    with suppress(BaseException):
        logger.error(
            "Rollout session cleanup failed while preserving the primary "
            "rollout error",
            exc_info=(
                type(cleanup_error),
                cleanup_error,
                cleanup_error.__traceback__,
            ),
        )


@contextmanager
def _rollout_session(actor_rollout_wg):
    begin = getattr(actor_rollout_wg, "begin_rollout_session", None)
    end = getattr(actor_rollout_wg, "end_rollout_session", None)
    if not callable(begin) or not callable(end):
        yield
        return

    primary_error = None
    try:
        begin()
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            end()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            _record_session_cleanup_error(primary_error, cleanup_error)


def _select_observation_rows(
    obs: Dict,
    indices: np.ndarray,
) -> Dict:
    selected = {}
    for key, value in obs.items():
        if value is None:
            selected[key] = None
        elif isinstance(value, torch.Tensor):
            selected[key] = value[
                torch.as_tensor(indices, device=value.device)
            ]
        elif isinstance(value, np.ndarray):
            selected[key] = value[indices]
        else:
            selected[key] = [value[index] for index in indices]
    return selected


def _scatter_observation_rows(
    obs: Dict,
    selected_obs: Dict,
    indices: np.ndarray,
) -> Dict:
    scattered = {}
    for key, value in obs.items():
        selected_value = selected_obs.get(key)
        if value is None or selected_value is None:
            scattered[key] = value
        elif isinstance(value, torch.Tensor):
            updated = value.clone()
            updated[
                torch.as_tensor(indices, device=value.device)
            ] = selected_value
            scattered[key] = updated
        elif isinstance(value, np.ndarray):
            updated = value.copy()
            updated[indices] = selected_value
            scattered[key] = updated
        else:
            updated = list(value)
            for selected_index, original_index in enumerate(indices):
                updated[original_index] = selected_value[selected_index]
            scattered[key] = updated
    return scattered


def _gamefiles_from_infos(infos: List[Dict]) -> np.ndarray:
    """Extract per-env AlfWorld gamefile paths from env step/reset infos."""
    return np.array([info.get("extra.gamefile") for info in infos], dtype=object)


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_gamefiles = obs.get('gamefile', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print("Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        # AlfWorld SDAR/RLSD: pass gamefile so teacher can inject task-specific skills.
        if obs_gamefiles is not None:
            gamefile = obs_gamefiles[item]
            if gamefile is not None:
                row_dict['gamefile'] = gamefile if isinstance(gamefile, str) else str(gamefile)

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def _prepare_and_generate_batch(
        self,
        gen_batch: DataProto,
        obs: Dict,
        actor_rollout_wg,
        original_indices=None,
    ) -> DataProto:
        batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
        if original_indices is not None:
            if "index" in batch.batch:
                index_values = batch.batch["index"]
                batch.batch["index"] = torch.as_tensor(
                    original_indices,
                    dtype=index_values.dtype,
                    device=index_values.device,
                )
            elif "index" in batch.non_tensor_batch:
                batch.non_tensor_batch["index"] = np.asarray(
                    original_indices,
                    dtype=batch.non_tensor_batch["index"].dtype,
                )

        non_tensor_batch_keys = ["raw_prompt_ids"]
        for key in (
            "multi_modal_data",
            "raw_prompt",
            "tools_kwargs",
        ):
            if key in batch.non_tensor_batch:
                non_tensor_batch_keys.append(key)

        batch_input = batch.pop(
            batch_keys=[
                "input_ids",
                "attention_mask",
                "position_ids",
            ],
            non_tensor_batch_keys=non_tensor_batch_keys,
        )
        batch_input.meta_info = gen_batch.meta_info
        batch_input, pad_size = pad_dataproto_to_divisor(
            batch_input,
            actor_rollout_wg.world_size,
        )
        batch_output = actor_rollout_wg.generate_sequences(batch_input)
        return batch.union(
            unpad_dataproto(batch_output, pad_size=pad_size)
        )


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for step_idx, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # turn_step: which step within the trajectory
                    data['turn_step'] = step_idx
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        current_gamefiles = _gamefiles_from_infos(infos)

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        with _rollout_session(actor_rollout_wg):
            for _step in range(self.config.env.max_steps):
                active_masks = np.logical_not(is_done)

                if hasattr(envs, "step_selected"):
                    active_indices = np.flatnonzero(active_masks)
                    active_gen_batch = gen_batch[active_indices]
                    obs_input = {
                        **obs,
                        'gamefile': current_gamefiles,
                    }
                    active_obs = _select_observation_rows(
                        obs_input,
                        active_indices,
                    )
                    batch = self._prepare_and_generate_batch(
                        active_gen_batch,
                        active_obs,
                        actor_rollout_wg,
                        original_indices=active_indices,
                    )

                    batch.non_tensor_batch['uid'] = (
                        uid_batch[active_indices]
                    )
                    batch.non_tensor_batch['traj_uid'] = (
                        traj_uid[active_indices]
                    )

                    text_actions = self.tokenizer.batch_decode(
                        batch.batch['responses'],
                        skip_special_tokens=True,
                    )
                    next_active_obs, rewards, dones, infos = (
                        envs.step_selected(
                            text_actions,
                            active_indices.tolist(),
                        )
                    )
                    current_gamefiles[active_indices] = (
                        _gamefiles_from_infos(infos)
                    )

                    if len(rewards.shape) == 2:
                        rewards = rewards.squeeze(1)
                    if len(dones.shape) == 2:
                        dones = dones.squeeze(1)

                    if 'is_action_valid' in infos[0]:
                        batch.non_tensor_batch['is_action_valid'] = (
                            np.array(
                                [
                                    info['is_action_valid']
                                    for info in infos
                                ],
                                dtype=bool,
                            )
                        )
                    else:
                        batch.non_tensor_batch['is_action_valid'] = (
                            np.ones(
                                len(active_indices),
                                dtype=bool,
                            )
                        )

                    if 'tool_calling' in infos[0]:
                        tool_callings[active_indices] += np.array(
                            [
                                info['tool_calling']
                                for info in infos
                            ],
                            dtype=np.float32,
                        )
                    episode_rewards[active_indices] += (
                        torch_to_numpy(rewards)
                    )
                    episode_lengths[active_indices] += 1

                    assert len(rewards) == len(active_indices), (
                        "env should return rewards for selected "
                        f"environments, got {len(rewards)} rewards for "
                        f"{len(active_indices)} environments"
                    )
                    batch.non_tensor_batch['rewards'] = (
                        torch_to_numpy(rewards, is_object=True)
                    )
                    batch.non_tensor_batch['active_masks'] = np.full(
                        len(active_indices),
                        True,
                        dtype=object,
                    )

                    batch_list: list[dict] = to_list_of_dict(batch)
                    for selected_index, original_index in enumerate(
                        active_indices
                    ):
                        total_batch_list[original_index].append(
                            batch_list[selected_index]
                        )
                        total_infos[original_index].append(
                            infos[selected_index]
                        )

                    is_done[active_indices] = np.logical_or(
                        is_done[active_indices],
                        dones,
                    )
                    obs = _scatter_observation_rows(
                        obs,
                        next_active_obs,
                        active_indices,
                    )

                    if is_done.all():
                        break
                    continue

                obs_input = {
                    **obs,
                    'gamefile': current_gamefiles,
                }
                batch = self._prepare_and_generate_batch(
                    gen_batch,
                    obs_input,
                    actor_rollout_wg,
                )

                batch.non_tensor_batch['uid'] = uid_batch
                batch.non_tensor_batch['traj_uid'] = traj_uid

                text_actions = self.tokenizer.batch_decode(
                    batch.batch['responses'],
                    skip_special_tokens=True,
                )
                next_obs, rewards, dones, infos = envs.step(
                    text_actions
                )
                current_gamefiles = _gamefiles_from_infos(infos)

                if len(rewards.shape) == 2:
                    rewards = rewards.squeeze(1)
                if len(dones.shape) == 2:
                    dones = dones.squeeze(1)

                if 'is_action_valid' in infos[0]:
                    batch.non_tensor_batch['is_action_valid'] = (
                        np.array(
                            [
                                info['is_action_valid']
                                for info in infos
                            ],
                            dtype=bool,
                        )
                    )
                else:
                    batch.non_tensor_batch['is_action_valid'] = (
                        np.ones(batch_size, dtype=bool)
                    )

                if 'tool_calling' in infos[0]:
                    tool_callings[active_masks] += np.array(
                        [
                            info['tool_calling']
                            for info in infos
                        ],
                        dtype=np.float32,
                    )[active_masks]
                episode_rewards[active_masks] += (
                    torch_to_numpy(rewards)[active_masks]
                )
                episode_lengths[active_masks] += 1

                assert len(rewards) == batch_size, (
                    "env should return rewards for all environments, "
                    f"got {len(rewards)} rewards for "
                    f"{batch_size} environments"
                )
                batch.non_tensor_batch['rewards'] = torch_to_numpy(
                    rewards,
                    is_object=True,
                )
                batch.non_tensor_batch['active_masks'] = torch_to_numpy(
                    active_masks,
                    is_object=True,
                )

                batch_list: list[dict] = to_list_of_dict(batch)
                for i in range(batch_size):
                    total_batch_list[i].append(batch_list[i])
                    total_infos[i].append(infos[i])

                is_done = np.logical_or(is_done, dones)
                obs = next_obs
                if is_done.all():
                    break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches
        target_trajectories = (
            self.config.data.train_batch_size
            * self.config.env.rollout.n
        )
        while (
            len(total_batch_list) < target_trajectories
            and try_count < max_try_count
        ):

            if len(total_batch_list) > 0:
                print(
                    f"valid num={len(total_batch_list)} < target num="
                    f"{target_trajectories}. Keep generating... "
                    f"({try_count}/{max_try_count})"
                )
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )

            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output
