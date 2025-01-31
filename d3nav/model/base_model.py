import torch


class BaseModel(torch.nn.Module):
    def load_net(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        if path is None or not path:
            # No model to load
            return
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            print("Loading optimizer state dict")
            parameters = parameters["model"]

        # validate all keys in state_dict are present in self.state_dict()
        # for k in parameters:
        #     if k not in self.state_dict():
        #         raise Exception(
        #             "Loading: {self.__class__.__name__} state_dict does not \
        #                 contain key {k} when loading from {path}".format(
        #                     self=self, k=k, path=path
        #                 )
        #         )

        incompatible_keys = self.load_state_dict(
            parameters,
            strict=False,
        )
        print("incompatible_keys", incompatible_keys)

        del parameters
        torch.cuda.empty_cache()

    def get_device(
        self,
    ):
        try:
            return next(self.parameters()).device
        except Exception as ex:
            print("No device found, using CPU", ex)
            return torch.device("cpu")

    def print_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        non_trainable_params = total_params - trainable_params

        total_size = sum(
            p.numel() * p.element_size() for p in self.parameters()
        ) / (1024 * 1024)
        trainable_size = sum(
            p.numel() * p.element_size()
            for p in self.parameters()
            if p.requires_grad
        ) / (1024 * 1024)

        print(f"{'=' * 40}")
        print(f"Model: {self.__class__.__name__}")
        print(f"{'=' * 40}")
        print(f"Total parameters:      {total_params:,}")
        print(f"Trainable parameters:  {trainable_params:,}")
        print(f"Non-trainable params:  {non_trainable_params:,}")
        print(f"{'-' * 40}")
        print(f"Total size:            {total_size:.2f} MB")
        print(f"Trainable size:        {trainable_size:.2f} MB")
