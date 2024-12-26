import argparse

def get_parser():
    # Create the parser
    parser = argparse.ArgumentParser(description='Parse model training arguments.')
    # File argument, if this is provided, use the arguments in that file instead
    parser.add_argument('-f', '--file', type=str,
                        help='''Path to configuration file in ./config_files.
                        If this is used, other arguments will be ignored''')
    parser.add_argument('-o', '--out', type=str, default='unnamed',
                        help='Path to output image file in ./images')

    # Training arguments
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for the model')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for training the model')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='number of epochs for training the model')

    # Data arguments
    parser.add_argument('--p', type=int, default=97,
                        help='modulo number for addition')
    parser.add_argument('--k', type=int, default=2,
                        help='number of summands for addition')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='training sample fraction')
    parser.add_argument('--n_samples', type=int, default=0,
                        help='total sample number for training and validation')

    # Model arguments
    parser.add_argument('--model', type=str,
                        help='type of model to use')
    parser.add_argument('--d_model', type=int, default=128,
                        help='embedding dimension of the model')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of layers of the model')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of attention heads of the model')
    return parser
