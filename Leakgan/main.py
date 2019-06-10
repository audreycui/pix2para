import getopt
import sys

from colorama import Fore
from Leakgan import Leakgan

from config import Config
#added test and val options to "train_type" flag  

def set_gan(gan_name):
    config = Config()

    gans = dict()
    gans['leakgan'] = Leakgan
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan(config)
        #gan.vocab_size = 5000
        #gan.generate_num = 10000
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)



def set_training(gan, training_method):
    try:
        if training_method == 'real':
            gan_func = gan.train_real
        elif training_method == 'test':
            gan_func = gan.test
        elif training_method == 'val':
            gan_func = gan.val
        else:
            print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
            sys.exit(-3)
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + training_method + Fore.RESET)
        sys.exit(-3)
    return gan_func


def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, "hig:t:d:")

        opt_arg = dict(opts)
        if '-h' in opt_arg.keys():
            print('usage: python main.py -g <gan_type>')
            print('       python main.py -g <gan_type> -t <train_type>')
            print('       python main.py -g <gan_type> -t real -d <your_data_location>')
            print('       python main.py -g <gan_type> -t real -d <your_data_location> -i')
            sys.exit(0)
        if not '-g' in opt_arg.keys():
            print('unspecified GAN type, use Leakgan training only...')
            gan = set_gan('leakgan')
        else:
            gan = set_gan(opt_arg['-g'])
        if not '-t' in opt_arg.keys():
            gan.train_oracle()
        else:
            has_image = True
            if '-i' in opt_arg.keys():
                has_image = False
            gan_func = set_training(gan, opt_arg['-t'])
            if opt_arg['-t'] == 'real' and '-d' in opt_arg.keys():
                gan_func(opt_arg['-d'], has_image)
            else:
                gan_func(data_loc=None, with_image=has_image)
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)
    pass


if __name__ == '__main__':
    gan = None
    parse_cmd(sys.argv[1:])
