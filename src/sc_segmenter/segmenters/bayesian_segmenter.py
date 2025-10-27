"""Perform the segmentation using a Bayesian approach relying on the PYLM
approach.
"""

from nhpylm.models import NHPYLMModel
from sc_segmenter.segmenters.segmenter import Segmenter


class BayesianSegmenter(Segmenter):
    """Initiate a Bayesian Segmenter using PYLM
    approach."""

    def __init__(self,
                 n_dim=7,
                 init_d=0.5,
                 init_theta=2.0,
                 init_a=6.0,
                 init_b=0.83333333,
                 beta_stops=1.0,
                 beta_passes=1.0,
                 d_a=1.0,
                 d_b=1.0,
                 theta_alpha=1.0,
                 theta_beta=1.0) -> None:
        """Init an object of class Bayesian Segmenter.

        Args:
            n_dim (int, optional): _description_. Defaults to 7.
            init_d (float, optional): _description_. Defaults to 0.5.
            init_theta (float, optional): _description_. Defaults to 2.0.
            init_a (float, optional): _description_. Defaults to 6.0.
            init_b (float, optional): _description_. Defaults to 0.83333333.
            beta_stops (float, optional): _description_. Defaults to 1.0.
            beta_passes (float, optional): _description_. Defaults to 1.0.
            d_a (float, optional): _description_. Defaults to 1.0.
            d_b (float, optional): _description_. Defaults to 1.0.
            theta_alpha (float, optional): _description_. Defaults to 1.0.
            theta_beta (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        super().__init__()
        self.model = NHPYLMModel(n_dim,
                                 init_d=init_d,
                                 init_theta=init_theta,
                                 init_a=init_a,
                                 init_b=init_b,
                                 beta_stops=beta_stops,
                                 beta_passes=beta_passes,
                                 d_a=d_a, d_b=d_b,
                                 theta_alpha=theta_alpha,
                                 theta_beta=theta_beta)

    def segment(self, text: str) -> list[str]:
        print("Segment to predict")
        print(text)
        print(self.model.predict_segments([text]))
        return self.model.predict_segments([text])[0][0]

    def train(self,
              train_x: list[str],
              dev_x: list[str],
              epochs: int = 20,
              d_theta_learning: bool = True,
              poisson_learning: bool = True,
              print_each_nth_iteration: int = 5):
        """Perform the training of the Bayesian model."""
        self.model.train(train_x,
                         dev_x,
                         epochs,
                         d_theta_learning,
                         poisson_learning,
                         print_each_nth_iteration)
