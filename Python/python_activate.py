import argparse
import sys
import BasicTriangleCount
import Naive


def main():
    parser = argparse.ArgumentParser(description='Python script to conduct triangle counting on GPU and CPU.')
    parser.add_argument('-g', '--graph_file', help="Path to the graph edge list file.", default=None)
    parser.add_argument('-m', '--mode', help="Define CPU or GPU algorithm. Default is CPU", default="CPU")
    parser.add_argument('-a', '--algorithm', help="Define what algorithm to use. Matrix_Multiplication or Naive. Default is Naive", default="Naive")

    args = parser.parse_args()

    graph_file = args.graph_file
    mode = args.mode
    algorithm = args.algorithm

    if graph_file and mode and algorithm:
        print(f"Running Graph located at {graph_file} with {algorithm} and {mode} mode.")
        if algorithm == "Matrix_Multiplication":
            BasicTriangleCount.basicTriangleCount.begin(mode, graph_file)
        elif algorithm == "Naive":
            Naive.NaiveTriangleCount.begin(mode, graph_file)
        else:
            print("Incorrect algorithm type given.", file=sys.stderr)
            raise ValueError


if __name__ == '__main__':
    main()