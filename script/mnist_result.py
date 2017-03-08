import argparse
import pandas as pd
import urllib2
from mnist_pytorch import *


# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')

parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--reload_model', action='store_true', default=True)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
test_batch_size = 100


def predict_labels(Q, X):
    Q.eval()
    X = X * 0.3081 + 0.1307
    X.resize_(test_batch_size, X_dim)
    X = Variable(X)
    if args.cuda:
        X = X.cuda()

    output = Q(X)[0]
    pl = output.data.max(1)[1].cpu().numpy().reshape(-1)
    return pl


def main():
    model = Q_net()
    if args.cuda:
        model = model.cuda()

    if args.reload_model:
        response = urllib2.urlopen('https://github.com/fducau/DSGA-1008-Spring2017-A1/blob/dual_train/script/Q_net?raw=true')
        Q = response.read()
        f = open('./Q_network', 'wb')
        f.write(Q)
        f.close()
    else:
        generate_model(saveto='./')

    model.load_state_dict(torch.load('./Q_network'))

    print('loading data!')
    data_path = '../data/'
    test_set = pickle.load(open(data_path + "test.p", "rb"))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

    label_predict = np.array([])
    model.eval()
    for data, target in test_loader:
        temp = predict_labels(model, data)
        label_predict = np.concatenate((label_predict, temp))

    prediction_df = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    prediction_df.reset_index(inplace=True)
    prediction_df.rename(columns={'index': 'ID'}, inplace=True)

    prediction_df.to_csv('sudaquian_submission.csv', index=False)
    print('Prediction saved')

if __name__ == '__main__':
    main()