import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()


    def initialize(self):
        self.parser.add_argument('--name', type=str, required=True, help="name of your result")
        self.parser.add_argument('--descriptions',type=str,default='',help="face descriptions")
        self.parser.add_argument('--classfier_path',type=str,default='./checkpoints/onehot_classfier/latest_parser.pth')
        self.parser.add_argument('--ShapeNet_path',type=str,default='./checkpoints/shape_synthesis/latest_shape.pth')
        self.parser.add_argument('--TextureNet_path',type=str,default='./checkpoints/texture_synthesis/latest_texture.pkl')
        self.parser.add_argument('--prompt',type=str,default='',help="face descriptions")
        self.parser.add_argument('--lr_latent',type=float,default=0.008,help="lr_latent")
        self.parser.add_argument('--lr_param',type=float,default=0.003,help="lr_param")
        self.parser.add_argument('--lambda_latent',type=float,default=0.0003,help="lambd_latent")
        self.parser.add_argument('--lambda_param',type=float,default=3,help="lambda_param")

        self.parser.add_argument('--save_step', type=int, default=5, help="save step")
        self.parser.add_argument('--step',type=int,default=100,help="all step")
        # self.parser.add_argument('--concrete_dir',type=str,default="./result/concrete_synthesis/",help="concrete save path")
        # self.parser.add_argument('--prompt_dir',type=str,default="./result/prompt_synthesis/",help="prompt save path")
        self.parser.add_argument('--result_dir',type=str,default="./result/final_result/",help="result save path")
        self.parser.add_argument('--inter_dir',type=str,default="./result/inter_result/",help="intermediate result")



    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt
