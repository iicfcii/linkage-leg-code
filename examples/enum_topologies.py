import sys
from linkage_leg.topology import enum


def main():
    if len(sys.argv) == 1:
        enum("logs")
    else:
        enum(sys.argv[1], resume=True)


if __name__ == "__main__":
    main()
