import options_ucf
class Config(object):
    def __init__(self, args):
        self.root_dir = args.root_dir
        self.modal = args.modal
        # self.lr = eval(args.lr)
        self.lr = args.lr
        self.num_iters = 5000
        self.len_feature = 1024 
        self.batch_size = args.batch_size
        self.model_path = args.model_path
        self.output_path = args.output_path
        self.num_workers = args.num_workers
        self.model_file = args.model_file
        self.seed = args.seed
        self.num_segments = args.num_segments

#
# if __name__ == "__main__":
#     args=options.parse_args()
#     conf=Config(args)
#     print(conf.lr)

