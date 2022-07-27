import argparse
import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.rating_prediction.W_autorec import * 
from utils.load_data.load_data_rating_simulation import * 


def parse_args():
    parser = argparse.ArgumentParser(description='nnRec')
    parser.add_argument('--model', default='I-AutoRec')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=256) 
    parser.add_argument('--learning_rate', type=float, default=1e-3)  
    parser.add_argument('--reg_rate', type=float, default=0.1)  
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=1000)
    parser.add_argument('--show_time', type=bool, default=False)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--deep_layers', type=str, default="200, 200, 200")
    parser.add_argument('--field_size', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    show_time = args.show_time,

    kws = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'reg_rate': reg_rate,
        'num_factors': num_factors,
        'display_step': display_step,
        'show_time': show_time[0],
        'T': args.T,
        'layers': list(map(int, args.deep_layers.split(','))),
        'field_size': args.field_size
    }

    train_data, vali_data, test_data,test_data_new, n_user, n_item =load_data_rating(path="../data/full.data890.csv",
                                                                                   header=0, seed = 1, n_new=50,
                                                                                   vali_size=0.15, test_size=0.1, sep=",") 

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    
    with tf.Session(config=config) as sess:
        model = None
        # Model selection
        if args.model == "I-AutoRec":
            model = IAutoRec(sess, n_user, n_item)
 
        # build and execute the model
        if model is not None:
            model.build_network()
            model.execute(train_data, vali_data, test_data, test_data_new)

