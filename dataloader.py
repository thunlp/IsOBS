from torchvision import transforms as T

transform = T.Compose([
    T.Resize([50,50]),
    # T.Grayscale(),
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])