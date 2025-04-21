from .dataset import MathDataset
import torch
from torchvision import transforms as TT
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .config import config

def data_loader():

    image_transform = TT.Compose([
        # Transform image to grayscale, resize, and normalize
        TT.Grayscale(num_output_channels=1),
        TT.Resize((256, 512)),
        TT.ToTensor(),
        TT.Normalize(mean=[0.5], std=[0.5])
    ])

    def target_transform(label):
        # Tokenize the label with start and end tokens
        tokenizer = Tokenizer.from_file(f"handwritten_math_solutions/src/latex_tokenizer.json")
        encoded_label = tokenizer.encode(f'<s>{label}</s>')
        return torch.LongTensor(encoded_label.ids)


    train_dataset = MathDataset(path='handwritten_math_solutions/src/data', phase="train", image_transform=image_transform, target_transform=target_transform)
    valid_dataset = MathDataset(path='handwritten_math_solutions/src/data', phase="valid", image_transform=image_transform, target_transform=target_transform)
    test_dataset = MathDataset(path='handwritten_math_solutions/src/data', phase="test", image_transform=image_transform, target_transform=target_transform)


    def collate_fn(data):
        # Load the tokenizer and get the padding token value
        token = Tokenizer.from_file(f"handwritten_math_solutions/src/latex_tokenizer.json")
        pad_value = token.get_vocab().get('<pad>', 3)

        tensors, targets = zip(*data)

        # Convert targets into tensors
        targets = [torch.tensor(t, dtype=torch.long) for t in targets]

        # Pad sequences to ensure equal length within a batch
        features = pad_sequence(targets, padding_value=pad_value, batch_first=True)

        try:
            # Stack tensors to create a batch
            tensors = torch.stack(tensors)
        except RuntimeError as e:
            # Print an error message if stacking fails
            print(f"stack error: {e}")
            for i, t in enumerate(tensors):
                print(f"tensor shape error {i}: {t.shape}")

        return tensors, features


    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, **config['data'])
    valid_loader = DataLoader(valid_dataset, shuffle=False, collate_fn=collate_fn, **config['data'])
    test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, **config['data'])
    return train_loader, valid_loader, test_loader