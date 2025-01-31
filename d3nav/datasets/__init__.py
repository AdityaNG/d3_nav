import torchvision

default_img_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # Convert to float
        torchvision.transforms.Lambda(lambda x: x.float()),
        # Normalize the image
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        # Unsqueeze to add a batch dimension
        torchvision.transforms.Lambda(lambda x: x.unsqueeze(0)),
    ]
)

inv_default_img_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
        ),
        # Convert to uint8
        torchvision.transforms.Lambda(lambda x: (x.clamp(0, 1) * 255).byte()),
        # Permute to (H, W, C)
        torchvision.transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        # Squueze to remove the batch dimension
        torchvision.transforms.Lambda(lambda x: x.squeeze(0)),
        # To numpy
        torchvision.transforms.Lambda(lambda x: x.detach().cpu().numpy()),
    ]
)
