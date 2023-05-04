import PIL.Image
import trimesh
import options
import torch
import numpy as np
from model import Classify_Network,Shape_Network,Texture_Network
import os
import clip
import dnnlib
import cv2
import torchvision
from torchvision.transforms.transforms import ToPILImage
import legacy
from torch import optim
from tqdm import tqdm
import pyredner
from pyredner.load_obj import load_mtl


#to replace trimesh.load
def load_ori_mesh(fn):
    return trimesh.load(fn,resolver=None,split_object=False,group_material=False,skip_materials=False,maintain_order=True,process=False)


class CLIPLoss(torch.nn.Module):
    def __init__(self, stylegan_size=512):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image, text):
        image_upsam = self.upsample(image)
        image = self.avg_pool(image_upsam)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

class generation(torch.nn.Module):
    def __init__(self):
        super(generation, self).__init__()

def encode_text(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    tokens = clip.tokenize(text,truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens).float()
    return text_features

def gen_onehot(opt,text):

    classify_model = Classify_Network.MLP()
    state = torch.load(opt.classfier_path)
    classify_model.load_state_dict(state['MLP'])
    classify_model.cuda()
    classify_model.eval()

    text_features = encode_text(text)

    onehot_pred = classify_model(text_features).reshape(24,8)
    shape_onehot = onehot_pred[:16,:]
    shape_onehot = ((shape_onehot==shape_onehot.max(dim=-1,keepdim=True)[0])*1).reshape(1,-1)
    texture_onehot = torch.cat((onehot_pred[:3,:],onehot_pred[16:,:]),dim=0)
    texture_onehot = ((texture_onehot==texture_onehot.max(dim=-1,keepdim=True)[0])*1).reshape(1,-1)
    all_pred = torch.cat((shape_onehot.reshape(-1,8),texture_onehot.reshape(-1,8)[3:,:]),dim=0)

    return all_pred,shape_onehot,texture_onehot

def gen_shape(shape_label,opt):

    device = torch.device("cuda:0")
    model = Shape_Network.MLP()
    state = torch.load(opt.ShapeNet_path)
    model.load_state_dict(state['MLP'])
    model.to(device)
    model.eval()
    mean_mesh = load_ori_mesh("./predef/mean_face_3DMM_300.obj")
    # mean_verts = np.load("./predef/mean_verts.npy")
    core = np.load("./predef/core_1627_300_weight_10.npy")
    pred_param = model(shape_label).detach().cpu().numpy()
    pred_verts = np.matmul(pred_param,core).reshape(-1,3)
    curr_mesh = mean_mesh.copy()
    curr_mesh.vertices = mean_mesh.vertices + pred_verts

    # curr_mesh.export("./result/0_1.obj");

    return curr_mesh,torch.from_numpy(pred_param).cuda()

def gen_texture(opt,texture_label):
    trans_pil = ToPILImage()
    device = torch.device('cuda')
    with dnnlib.util.open_url(opt.TextureNet_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    Mapping = G.mapping
    Synthesis = G.synthesis
    z = torch.randn(1, G.z_dim).to(device)
    ws = Mapping(z, texture_label)
    img = Synthesis(ws, noise_mode='const')
    texture = torch.clip((img[0]+1)/2,0,1)
    # texture = texture.squeeze(0)  # 压缩一维
    texture = trans_pil(texture)
    # texture.save("./result/material_0.png")

    return texture,ws,Synthesis

def concrete_synthesis(opt,shape_label,texture_label):

    mesh,pred_param = gen_shape(shape_label,opt)
    texture,ws,Synthesis = gen_texture(opt,texture_label=texture_label)

    mesh.visual.material.image = texture

    save_path = os.path.join(opt.result_dir,opt.name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    obj_save_path = os.path.join(save_path,"result_concrete.obj")
    mesh.export(obj_save_path);

    return ws,pred_param,Synthesis


def diff_render(render_img,curr_verts):
    device = "cuda"
    mean_verts = torch.from_numpy(np.load("./predef/mean_verts.npy")).cuda()
    m = load_mtl("../predef/material_0.mtl")['material_0']
    spec = torch.tensor(m.Ks, dtype=torch.float32, device=device)
    rough = torch.tensor([2.0 / (m.Ns + 2.0)], dtype=torch.float32, device=device)
    faces = torch.from_numpy(np.load("./predef/faces.npy")).to(torch.int32).cuda()
    uv = np.load("./predef/uv.npy")
    for i in range(uv.shape[0]):
        uv[i][1] = 1 - uv[i][1]
    uv = torch.from_numpy(uv).to(torch.float32).cuda()

    m = pyredner.Material(diffuse_reflectance=(render_img[0] + 1) / 2,
                          specular_reflectance=spec,
                          roughness=rough)

    obj_pred = pyredner.Object(vertices=curr_verts + mean_verts, indices=faces, uvs=uv, material=m)
    cam_pred = pyredner.automatic_camera_placement([obj_pred], resolution=(512, 512))
    cam_pred.look_at[-1] = -cam_pred.look_at[-1]
    cam_pred.position[-1] = -cam_pred.position[-1]
    curr_scene_pred = pyredner.Scene(camera=cam_pred, objects=[obj_pred])
    img_pred = pyredner.render_albedo(curr_scene_pred)

    return img_pred


def prompt_synthesis(ws,params,Synthesis):
    trans_pil = ToPILImage()
    latent_code_int = ws
    param_init = params

    latent = latent_code_int.detach().clone()
    latent.requires_grad = True
    param = param_init.detach().clone()
    param.requires_grad = True
    core = torch.from_numpy(np.load("./predef/core_1627_300_weight_10.npy")).cuda()

    clip_loss = CLIPLoss()

    latent_optimizer = optim.Adam([latent], lr=opt.lr_latent)
    params_optimizer = optim.Adam([param], lr=opt.lr_param)

    pbar = tqdm(range(opt.step))

    for i in pbar:

        img_gen = Synthesis(latent)  ## 1*3*512*512

        curr_verts = torch.matmul(param,core).reshape(-1,3)

        render_img = img_gen.permute(0,2,3,1)

        ## rendering only the front image is enough to generate reasonable results and can save lots of time.
        img_pred = diff_render(render_img,curr_verts)
        img_rgb = img_pred.detach().cpu().numpy()[:,:,[2,1,0]]*255
        img_chw = img_pred.unsqueeze(0).permute(0,3,1,2)

        text_tokens = clip.tokenize(opt.prompt).cuda()
        c_loss = clip_loss(img_chw,text_tokens)
        l2_loss_latent = ((latent-latent_code_int)**2).sum()
        l2_loss_param = ((param-params)**2).sum()

        loss = c_loss + opt.lambda_latent * l2_loss_latent + opt.lambda_param * l2_loss_param

        latent_optimizer.zero_grad()
        params_optimizer.zero_grad()
        loss.backward()
        latent_optimizer.step()
        params_optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )

        if opt.save_step > 0 and i % opt.save_step == 0:
            with torch.no_grad():
                img_gen = Synthesis(latent)

            torchvision.utils.save_image(img_gen,f"{opt.inter_dir}/texture_map/{str(i).zfill(5)}_tex.jpg", normalize=True, range=(-1, 1))
            cv2.imwrite(os.path.join(opt.inter_dir,"render",f"{str(i).zfill(5)}_render.jpg"),img_rgb)

    img_final = Synthesis(latent)
    # texture_final = PIL.Image.fromarray(np.clip(np.uint8(((img_final[0].permute(1,2,0).detach().cpu().numpy()+1)/2)*255),0,255))
    texture_final = trans_pil(torch.clip((img_final[0]+1)/2,0,1))


    verts_final = torch.matmul(param,core).reshape(-1,3).detach().cpu().numpy()
    mean_mesh = load_ori_mesh("./predef/mean_face_3DMM_300.obj")
    final_mesh = mean_mesh.copy()
    final_mesh.visual.material.image = texture_final
    final_mesh.vertices = mean_mesh.vertices + verts_final
    curr_save_folder = os.path.join(opt.result_dir,opt.name,opt.prompt)
    if not os.path.exists(curr_save_folder):
        os.mkdir(curr_save_folder)

    final_mesh.export(os.path.join(curr_save_folder,"result_prompt.obj"));


def gen_full_mesh(opt):

    ## Text Parser: generate ont-hot code
    all_label,shape_label,texture_label = gen_onehot(opt,text=opt.descriptions)

    new_map_dict = {'race': ['Asian', 'Western people', 'Black people'],
                    'gender': ['male', 'female'],
                    'age': ['child', 'young', 'middle-aged', 'old'],
                    'eye_shape': ['almond', 'round', 'drooping', 'upturned', 'squinted', 'triangle', 'slender',
                                  'sunken'],
                    'eye_distance': ['wide', 'narrow', 'medium width'],
                    'eye_size': ['big', 'small', 'medium sized'],
                    'eyelid': ['single', 'double'],
                    'nose_size': ['big', 'small', 'medium sized'],
                    'nose_height': ['high', 'low', 'medium height'],
                    'nasal_base_shape': ['straight', 'upturned', 'lowered'],
                    'nose_width': ['wide', 'narrow', 'medium width'],
                    'mouth_width': ['wide', 'narrow', 'medium width'],
                    'lip_thickness': ['thick', 'thick upper', 'thick lower', 'thin', 'medium thickness'],
                    'lip_shape': ['round', 'bow-shaped', 'heart-shaped', 'downward-turned'],
                    'face_shape': ['oval', 'square', 'round', 'diamond', 'heart shaped', 'long'],
                    'face': ['fat', 'thin', 'medium'],
                    'pupil_color': ['black', 'brown', 'blue', 'amber', 'green', 'gray'],
                    'eyebrow_shape': ['round', 'large angle', 'flat', 'small angle', 'S-shaped', 'sword-shaped'],
                    'eyebrow_color': ['black', 'brown', 'gray'],
                    'eyebrow_density': ['dense', 'sparse', 'medium'],
                    'beard_yn': ['y', 'n'],
                    'beard_density': ['dense', 'medium', 'sparse','none'],
                    'beard_shape': ['moustache', 'stubble', 'whisker', 'beard', 'other beard','none'],
                    'beard_color': ['black', 'gray','none']
                    }

    for i,key in enumerate(new_map_dict.keys()):
        index = list(all_label[i,:]).index(1.0)
        if index < len(new_map_dict[key]):
            print(key,new_map_dict[key][index])
        else:
            print(key,"error")

    ## Concrete Synthesis
    ws, pred_param, Synthesis = concrete_synthesis(opt,shape_label,texture_label)

    ## Abstract Synthesis
    if opt.prompt:
        prompt_synthesis(ws,pred_param,Synthesis)


if __name__ == '__main__':
    opt = options.Options().parse()
    gen_full_mesh(opt)



