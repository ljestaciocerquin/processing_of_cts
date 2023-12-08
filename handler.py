import os
import json
from   train_denoiseCT import Train_DenoiseCT


class Handler(object):
    def __init__(self, args):
        self.args       = args
        self.task       = args.task
        self.output_dir = args.output_dir
        self.save_config_file()
        
    def save_config_file(self):
        exp_dir = self.output_dir + self.task 
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
            
        file_name = os.path.join(exp_dir, 'config.txt')
        with open(file_name, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
                

    def train_denoiseCT(self):
        self.train = Train_DenoiseCT(args=self.args)
        self.train.run_train()
    
    '''def test_registration_baseline(self):
        self.eval = Evaluation(args=self.args)
        self.eval.run_eval()'''
    
    
    def visualization(self):
        pass

    
    def run(self ):
        if self.task == 'train':
            print("Training!!!!")
            self.train_denoiseCT()
            
        elif self.task == 'test':
            print('Evaluation!!!!')
            #self.test_registration_baseline()
        elif self.task == 'vis':
            print('Under development...!')
        else:
            raise NotImplementedError('undefined task!')
