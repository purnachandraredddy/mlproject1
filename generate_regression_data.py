import argparse
import csv
import numpy as np

def linear_data_generator(m, b, rnge, N, scale, seed):
    rng = np.random.default_rng(seed=seed)
    sample = rng.uniform(low=rnge[0], high=rnge[1], size=(N, m.shape[0]))
    ys = np.dot(sample, np.reshape(m, (-1, 1))) + b
    noise = rng.normal(loc=0., scale=scale, size=ys.shape)
    return sample, ys + noise

def write_data(filename, X, y):
    with open(filename, "w") as file:
        xs = [f"x_{n}" for n in range(X.shape[1])]
        header = xs + ["y"]
        writer = csv.writer(file)
        writer.writerow(header)
        for row in np.hstack((X, y)):
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, help="Number of samples.")
    parser.add_argument("-m", nargs='*', type=float, help="Expected regression coefficients")
    parser.add_argument("-b", type=float, help="Offset")
    parser.add_argument("-scale", type=float, help="Scale of noise")
    parser.add_argument("-rnge", nargs=2, type=float, help="Range of Xs")
    parser.add_argument("-seed", type=int, help="A seed to control randomness")
    parser.add_argument("-output_file", type=str, help="Path to output file")
    args = parser.parse_args()
    m = np.array(args.m)
    X, y = linear_data_generator(m, args.b, args.rnge, args.N, args.scale, args.seed)
    write_data(args.output_file, X, y)

if __name__ == "__main__":
    main()
