"""Unit tests for memory.replay_buffer — all three buffer types and factory."""
import numpy as np
import pytest
from memory.replay_buffer import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    NstepReplayBuffer,
    create_buffer,
)
from tests.helpers import make_transitions, fill_buffer


class TestReplayBuffer:

    def test_push_increments_length(self):
        buf = ReplayBuffer(100)
        transitions = make_transitions(5)
        for t in transitions:
            buf.push(*t)
        assert len(buf) == 5

    def test_max_capacity_not_exceeded(self):
        buf = ReplayBuffer(10)
        fill_buffer(buf, 20)
        assert len(buf) == 10

    def test_sample_returns_correct_shapes(self):
        buf = ReplayBuffer(100)
        fill_buffer(buf, 20)
        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.shape == (8, 4)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 4)
        assert dones.shape == (8,)

    def test_update_priorities_is_noop(self):
        buf = ReplayBuffer(100)
        fill_buffer(buf, 20)
        # Should not raise and should not change length
        buf.update_priorities([0, 1, 2], [0.5, 0.3, 0.7])
        assert len(buf) == 20

    def test_mean_priority_returns_zero(self):
        buf = ReplayBuffer(100)
        fill_buffer(buf, 10)
        assert buf.mean_priority() == pytest.approx(0.0)

    def test_push_accepts_td_error_keyword(self):
        buf = ReplayBuffer(100)
        t = make_transitions(1)[0]
        buf.push(*t, _td_error=0.5)  # Should not raise
        assert len(buf) == 1

    def test_sample_float32_rewards(self):
        buf = ReplayBuffer(100)
        fill_buffer(buf, 20)
        _, _, rewards, _, dones = buf.sample(8)
        assert rewards.dtype == np.float32
        assert dones.dtype == np.float32


class TestPrioritizedReplayBuffer:

    def test_push_increments_size(self):
        buf = PrioritizedReplayBuffer(100)
        fill_buffer(buf, 10)
        assert len(buf) == 10

    def test_push_with_td_error_sets_priority(self):
        buf = PrioritizedReplayBuffer(100, eps=1e-6)
        t = make_transitions(1)[0]
        buf.push(*t, td_error=2.0)
        assert buf.priorities[0] == pytest.approx(2.0 + 1e-6, rel=1e-4)

    def test_push_without_td_error_uses_max_priority(self):
        buf = PrioritizedReplayBuffer(100)
        t = make_transitions(1)[0]
        buf.push(*t)
        assert buf.priorities[0] == buf.max_priority

    def test_capacity_overflow_wraps_position(self):
        buf = PrioritizedReplayBuffer(5)
        fill_buffer(buf, 10)
        assert len(buf) == 5

    def test_sample_returns_seven_elements(self):
        buf = PrioritizedReplayBuffer(100)
        fill_buffer(buf, 20)
        result = buf.sample(8)
        assert len(result) == 7

    def test_sample_is_weights_in_unit_interval(self):
        buf = PrioritizedReplayBuffer(100)
        fill_buffer(buf, 20)
        *_, is_weights = buf.sample(8)
        assert np.all(is_weights >= 0.0)
        assert np.all(is_weights <= 1.0 + 1e-6)

    def test_sample_indices_within_valid_range(self):
        buf = PrioritizedReplayBuffer(100)
        fill_buffer(buf, 20)
        *_, indices, _ = buf.sample(8)
        assert np.all(indices >= 0)
        assert np.all(indices < 20)

    def test_update_priorities_changes_stored_priorities(self):
        buf = PrioritizedReplayBuffer(100)
        fill_buffer(buf, 20)
        buf.update_priorities([0, 1], np.array([5.0, 3.0]))
        assert buf.priorities[0] == pytest.approx(5.0 + buf.eps, rel=1e-4)
        assert buf.priorities[1] == pytest.approx(3.0 + buf.eps, rel=1e-4)

    def test_mean_priority_positive(self):
        buf = PrioritizedReplayBuffer(100)
        fill_buffer(buf, 10)
        assert buf.mean_priority() > 0.0

    def test_uses_numpy_generator(self):
        buf = PrioritizedReplayBuffer(100)
        assert isinstance(buf.rng, np.random.Generator)


class TestNstepReplayBuffer:

    def test_push_single_episode_n_steps(self):
        buf = NstepReplayBuffer(100, n_step=3, gamma=0.99)
        transitions = make_transitions(5)
        for t in transitions:
            buf.push(*t)
        # With n=3 and 5 transitions (last done=True), buffer should have entries
        assert len(buf) > 0

    def test_nstep_return_accumulated(self):
        buf = NstepReplayBuffer(100, n_step=3, gamma=1.0)
        # Push 3 transitions with rewards [1.0, 2.0, 3.0], last is done
        state = np.zeros(4, dtype=np.float32)
        for i, reward in enumerate([1.0, 2.0, 3.0]):
            done = i == 2
            buf.push(state, 0, reward, state, done)
        # Full n-step return (all 3 steps) = 1+2+3 = 6.0 must be stored
        n = len(buf)
        assert n > 0
        _, _, rewards, _, _ = buf.sample(n)
        assert np.any(np.isclose(rewards, 6.0, rtol=1e-4))

    def test_done_flushes_buffer(self):
        buf = NstepReplayBuffer(100, n_step=5, gamma=0.99)
        state = np.zeros(4, dtype=np.float32)
        # Push 3 transitions and mark done — should flush even without reaching n=5
        for i in range(3):
            done = i == 2
            buf.push(state, 0, 1.0, state, done)
        assert len(buf) > 0

    def test_update_priorities_is_noop(self):
        buf = NstepReplayBuffer(100)
        state = np.zeros(4, dtype=np.float32)
        buf.push(state, 0, 1.0, state, True)
        buf.update_priorities([0], [0.5])  # Should not raise
        assert len(buf) > 0

    def test_mean_priority_returns_zero(self):
        buf = NstepReplayBuffer(100)
        state = np.zeros(4, dtype=np.float32)
        buf.push(state, 0, 1.0, state, True)
        assert buf.mean_priority() == pytest.approx(0.0)

    def test_sample_returns_five_elements(self):
        buf = NstepReplayBuffer(100, n_step=3, gamma=0.99)
        state = np.zeros(4, dtype=np.float32)
        # Fill with complete episodes
        for _ in range(5):
            for i in range(4):
                buf.push(state, 0, 1.0, state, i == 3)
        if len(buf) >= 4:
            result = buf.sample(4)
            assert len(result) == 5


class TestCreateBufferFactory:

    def test_creates_replay_buffer(self, small_config):
        small_config.buffer_type = "replay"
        buf = create_buffer(small_config)
        assert isinstance(buf, ReplayBuffer)

    def test_creates_prioritized_buffer(self, per_config):
        per_config.buffer_type = "prioritized"
        buf = create_buffer(per_config)
        assert isinstance(buf, PrioritizedReplayBuffer)

    def test_creates_nstep_buffer(self, small_config):
        small_config.buffer_type = "nstep"
        buf = create_buffer(small_config)
        assert isinstance(buf, NstepReplayBuffer)

    def test_unknown_type_raises(self, small_config):
        small_config.buffer_type = "unknown"
        with pytest.raises(ValueError, match="Unknown buffer_type"):
            create_buffer(small_config)

    def test_prioritized_alpha_passed(self, per_config):
        per_config.buffer_type = "prioritized"
        per_config.per_alpha = 0.7
        buf = create_buffer(per_config)
        assert buf.alpha == pytest.approx(0.7)
