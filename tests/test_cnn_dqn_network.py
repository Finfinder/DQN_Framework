"""Unit tests for models.cnn_dqn_network.CNNDQN — forward pass, CPU-only."""
import torch

from models.cnn_dqn_network import CNNDQN
from models.dqn_network import create_network


_SMALL_SHAPE = (4, 32, 32)
_ACTION_DIM = 6
_SMALL_CONV = [(8, 4, 2), (16, 3, 1)]
_SMALL_HIDDEN = 64

_PROD_SHAPE = (4, 84, 84)
_PROD_CONV = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]


class TestCNNDQNCreation:

    def test_default_conv_layers_creates_without_error(self):
        net = CNNDQN(_PROD_SHAPE, _ACTION_DIM)
        assert net is not None

    def test_action_dim_attribute(self):
        net = CNNDQN(_SMALL_SHAPE, _ACTION_DIM, conv_layers=_SMALL_CONV, hidden_dim=_SMALL_HIDDEN)
        assert net.action_dim == _ACTION_DIM

    def test_standard_variant_has_q_head(self):
        net = CNNDQN(_SMALL_SHAPE, _ACTION_DIM, conv_layers=_SMALL_CONV, hidden_dim=_SMALL_HIDDEN, dueling=False)
        assert hasattr(net, "q_head")
        assert not hasattr(net, "value_head")
        assert not hasattr(net, "advantage_head")

    def test_dueling_variant_has_value_and_advantage_heads(self):
        net = CNNDQN(_SMALL_SHAPE, _ACTION_DIM, conv_layers=_SMALL_CONV, hidden_dim=_SMALL_HIDDEN, dueling=True)
        assert hasattr(net, "value_head")
        assert hasattr(net, "advantage_head")
        assert not hasattr(net, "q_head")

    def test_dueling_attribute_false_by_default(self):
        net = CNNDQN(_SMALL_SHAPE, _ACTION_DIM, conv_layers=_SMALL_CONV, hidden_dim=_SMALL_HIDDEN)
        assert net.dueling is False

    def test_dueling_attribute_true_when_set(self):
        net = CNNDQN(_SMALL_SHAPE, _ACTION_DIM, conv_layers=_SMALL_CONV, hidden_dim=_SMALL_HIDDEN, dueling=True)
        assert net.dueling is True

    def test_custom_conv_layers_creates_without_error(self):
        net = CNNDQN(_SMALL_SHAPE, _ACTION_DIM, conv_layers=_SMALL_CONV, hidden_dim=_SMALL_HIDDEN)
        assert net is not None


class TestCNNDQNForwardStandard:

    def _make_net(self):
        net = CNNDQN(_SMALL_SHAPE, _ACTION_DIM, conv_layers=_SMALL_CONV, hidden_dim=_SMALL_HIDDEN, dueling=False)
        net.eval()
        return net

    def test_output_shape_batch_one(self):
        net = self._make_net()
        x = torch.zeros(1, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, _ACTION_DIM)

    def test_output_shape_batch_four(self):
        net = self._make_net()
        x = torch.zeros(4, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (4, _ACTION_DIM)

    def test_output_dtype_float32(self):
        net = self._make_net()
        x = torch.zeros(2, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert out.dtype == torch.float32

    def test_output_no_nan(self):
        net = self._make_net()
        x = torch.zeros(2, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert not torch.isnan(out).any()

    def test_output_no_inf(self):
        net = self._make_net()
        x = torch.zeros(2, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert not torch.isinf(out).any()

    def test_deterministic_in_eval_mode(self):
        net = self._make_net()
        x = torch.rand(2, *_SMALL_SHAPE)
        with torch.no_grad():
            out1 = net(x)
            out2 = net(x)
        assert torch.equal(out1, out2)

    def test_production_shape_84x84_default_conv(self):
        net = CNNDQN(_PROD_SHAPE, _ACTION_DIM, conv_layers=_PROD_CONV, hidden_dim=512, dueling=False)
        net.eval()
        x = torch.zeros(1, *_PROD_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, _ACTION_DIM)


class TestCNNDQNForwardDueling:

    def _make_net(self):
        net = CNNDQN(_SMALL_SHAPE, _ACTION_DIM, conv_layers=_SMALL_CONV, hidden_dim=_SMALL_HIDDEN, dueling=True)
        net.eval()
        return net

    def test_output_shape_batch_one(self):
        net = self._make_net()
        x = torch.zeros(1, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, _ACTION_DIM)

    def test_output_shape_batch_four(self):
        net = self._make_net()
        x = torch.zeros(4, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (4, _ACTION_DIM)

    def test_output_dtype_float32(self):
        net = self._make_net()
        x = torch.zeros(2, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert out.dtype == torch.float32

    def test_output_no_nan(self):
        net = self._make_net()
        x = torch.zeros(2, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert not torch.isnan(out).any()

    def test_output_no_inf(self):
        net = self._make_net()
        x = torch.zeros(2, *_SMALL_SHAPE)
        with torch.no_grad():
            out = net(x)
        assert not torch.isinf(out).any()

    def test_advantage_normalization_mean_near_zero(self):
        net = self._make_net()
        x = torch.rand(4, *_SMALL_SHAPE)
        with torch.no_grad():
            q_out = net(x)
            fc_out = net.fc(net.conv_trunk(x))
            value = net.value_head(fc_out)
            advantage = net.advantage_head(fc_out)
            expected = value + advantage - advantage.mean(dim=1, keepdim=True)
        assert torch.allclose(q_out, expected, atol=1e-5)

    def test_output_shape_matches_standard_variant(self):
        net_std = CNNDQN(_SMALL_SHAPE, _ACTION_DIM, conv_layers=_SMALL_CONV, hidden_dim=_SMALL_HIDDEN, dueling=False)
        net_duel = self._make_net()
        x = torch.zeros(3, *_SMALL_SHAPE)
        with torch.no_grad():
            out_std = net_std(x)
            out_duel = net_duel(x)
        assert out_std.shape == out_duel.shape


class TestCNNDQNFactory:

    def test_create_network_returns_cnndqn_instance(self, cnn_config):
        state_shape = (cnn_config.frame_stack, *cnn_config.frame_size)
        net = create_network(cnn_config, state_shape, _ACTION_DIM)
        assert isinstance(net, CNNDQN)

    def test_factory_forward_pass_output_shape(self, cnn_config):
        state_shape = (cnn_config.frame_stack, *cnn_config.frame_size)
        net = create_network(cnn_config, state_shape, _ACTION_DIM)
        net.eval()
        x = torch.zeros(2, *state_shape)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (2, _ACTION_DIM)

    def test_factory_dueling_variant_attribute(self, cnn_config):
        cnn_config.use_dueling = True
        state_shape = (cnn_config.frame_stack, *cnn_config.frame_size)
        net = create_network(cnn_config, state_shape, _ACTION_DIM)
        assert net.dueling is True
        assert hasattr(net, "value_head")
        assert hasattr(net, "advantage_head")

    def test_factory_standard_variant_attribute(self, cnn_config):
        cnn_config.use_dueling = False
        state_shape = (cnn_config.frame_stack, *cnn_config.frame_size)
        net = create_network(cnn_config, state_shape, _ACTION_DIM)
        assert net.dueling is False
        assert hasattr(net, "q_head")
