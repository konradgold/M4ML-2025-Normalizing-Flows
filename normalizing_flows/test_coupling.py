import unittest
import torch
from affine_coupling import Scale, Translation, Permute, AffineCoupling, NormalizingFlow


class TestAffineModules(unittest.TestCase):

    def test_scale_output_shape(self):
        layer = Scale(4, 4, hidden_size=32)
        x = torch.randn(2, 4)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_translation_output_shape(self):
        layer = Translation(3, 5, hidden_size=16)
        x = torch.randn(2, 3)
        y = layer(x)
        self.assertEqual(y.shape, (2, 5))

    def test_permute_forward_inverse(self):
        x = torch.randn(2, 5)
        perm = Permute(5)
        y, _ = perm(x)
        x_recovered = perm.inverse(y)
        self.assertTrue((x_recovered == x).all()) # Should not even change a bit

    def test_affine_forward_inverse_identity(self):
        coupling = AffineCoupling(size=6, d=3)
        x = torch.randn(4, 6)
        y, _ = coupling(x)
        x_recovered = coupling.inverse(y)
        self.assertTrue(torch.allclose(x, x_recovered, atol=1e-5))

        assert coupling.test_identity() # check built-in test

    def test_affine_log_det_shape(self):
        coupling = AffineCoupling(size=6, d=3)
        x = torch.randn(10, 6)
        y, log_det = coupling(x)
        self.assertEqual(log_det.shape, torch.Size([10]))

    def test_normalizing_flow_forward_inverse_identity(self):
        flow = NormalizingFlow(input_dim=8, num_layers=3)
        x = torch.randn(5, 8)
        z, _ = flow.forward_train(x)
        x_recovered = flow.inverse(z)
        assert flow.test_identity() # check built-in test
        self.assertTrue(torch.allclose(x, x_recovered, atol=1e-5))

    def test_normalizing_flow_log_det_shape(self):
        flow = NormalizingFlow(input_dim=6, num_layers=2)
        x = torch.randn(7, 6)
        _, log_det = flow.forward_train(x)
        self.assertEqual(log_det.shape, torch.Size([7]))


if __name__ == '__main__':
    unittest.main()