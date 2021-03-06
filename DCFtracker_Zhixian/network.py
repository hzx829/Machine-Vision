import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        # wf: the fourier transformation of correlation kernel w. You will need to calculate the best wf in update method.
        self.wf = None
        # xf: the fourier transformation of target patch x.
        self.xf = None
        self.config = config

    def mul_compl(self,z,w):
            real = z[...,0] * w[...,0] - z[...,1] * w[...,1]
            img  = z[...,0] * w[...,1] + z[...,1] * w[...,0]
            return torch.stack((real,img),-1)

    def mulconj_compl(self,z,w):
            real = z[...,0] * w[...,0] + z[...,1] * w[...,1]
            
            img  = z[...,1] * w[...,0] - z[...,0] * w[...,1]
            
            return torch.stack((real,img),-1)

    def forward(self, z):
        """
        :param z: the multiscale searching patch. Shape (3 , num_scale, crop_sz, crop_sz)
        :return response: the response of cross correlation. Shape (3 , 1, crop_sz, crop_sz)

        Calculate response using self.wf to do cross correlation on the searching patch z
        """
        # obtain feature of z and add hanning window
        z = self.feature(z) * self.config.cos_window
        

        
        

        z_fft = torch.rfft(z,signal_ndim=2)
                  response_f[:,num_s,m,n,:] += mulconj_compl(z_fft[l,num_s,m,n],self.wf[:,l,m,n,:])
       
        response_f = torch.sum(self.mulconj_compl(z_fft,self.wf),dim=1,keepdim=True).to(device)
        response = (torch.irfft(response_f,signal_ndim=2)).to(device)


        return response

    def update(self, x, lr=1.0):
        """
        this is the to get the fourier transformation of  optimal correlation kernel w
        :param x: the input target patch (1, 3, h ,w)
        :param lr: the learning rate to update self.xf and self.wf

        The other arguments concealed in self.config that will be used here:
        -- self.config.cos_window: the hanning window applied to the x feature. Shape (crop_sz, crop_sz),
                                   where crop_sz is 125 in default.
        -- self.config.yf: the fourier transform of idea gaussian response. Shape (1, 1, crop_sz, crop_sz//2+1, 2)
        -- self.config.lambda0: the coefficient of the normalize term.

        things you need to calculate:
        -- self.xf: the fourier transformation of x. Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        -- self.wf: the fourier transformation of optimal correlation filter w, calculated by the formula,
                    Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        """
        
        with torch.no_grad():
            # x: feature of patch x with hanning window. Shape (1, 32, crop_sz, crop_sz)
            x = self.feature(x) * self.config.cos_window
            #calculate self.xf and self.wf
            xf = torch.rfft(x,signal_ndim =2 )
            if lr == 1.0:
                self.xf = xf
                #xff = torch.sum(self.mulconj_compl(self.xf,self.xf) ,dim=1,keepdim=True)

                xff_lambda0 = torch.sum(torch.sum(self.xf ** 2 ,dim=4,keepdim=True) ,dim=1,keepdim=True)+self.config.lambda0
                wf = self.mulconj_compl( self.xf ,self.config.yf)/(xff_lambda0)
                self.wf = wf
                
            else:
                
                
                self.xf = (1-lr)*self.xf + lr*xf
                xff_lambda0 = torch.sum(torch.sum(self.xf ** 2 ,dim=4,keepdim=True) ,dim=1,keepdim=True) +self.config.lambda0
                
                wf = self.mulconj_compl( self.xf ,self.config.yf)/(xff_lambda0)
                self.wf = (1-lr)*self.wf + lr*wf
            



        







    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)

