import argparse

from . import modules
from . import utils
from . import dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model with specified parameters.")
    parser.add_argument("-t", "--type", type=str, required=True, help="Type is classification or regression.")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("-r", "--result",  type=str, required=True, help="Path to the result directory.")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="Number of training epochs. ")
    parser.add_argument("-l", "--learningrate",  type=float, default=0.00001, help="Learning rate for the optimizer")
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=[0], help="List of GPU device IDs to use for training.")
    return parser.parse_args()

    
def main():
    args = parse_args()
    print("--------")
    print(f"Task type: {args.type}")
    print(f"Dataset path: {args.path}")
    print(f"Result path: {args.result}")
    print(f"Epoch: {args.epoch}")
    print(f"Learning rate: {args.learningrate}")
    print(f"Using devices: {args.devices}")
    print("--------")
    
    device = utils.setup_device(args.devices)
    dataList, dataLabel, idTmps = dataset.process_data(args.path, args.type)
    trainloader = dataset.get_dataloader(dataList, dataLabel, idTmps, batch_size=1, shuffle=True)
    model_path = utils.train_model(args.type, args.learningrate, trainloader, device, args.epoch,args.result)
    
    df_attr = modules.calculate_attribution_scores(model_path, args.type, dataList, dataLabel, idTmps, args.path, args.result, device)
    utils.plot_attribution_scores(args.type, df_attr, args.result)
    
    df_attn = modules.calculate_attention_scores(model_path, trainloader, args.path, args.result, device)
    utils.plot_attention_scores(args.type, df_attn, args.path, args.result)
    
if __name__ == "__main__":
    main()