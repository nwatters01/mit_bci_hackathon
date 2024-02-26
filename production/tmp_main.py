import sys


def main():
    print('main')


if __name__ == '__main__':
    try:
        input('Start?')
        print('sup')
        sys.exit()
        main()
    except KeyboardInterrupt:
        print('\ninterrupted\n')
        sys.exit()
