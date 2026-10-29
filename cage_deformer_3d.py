import os
import json
import random
from release_utils import load_classifier, load_cage_checkpoint, classification_labels, attack_metrics
from pprint import pprint
import traceback
import sys
import shutil
import openmesh as om
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import pymesh
import torch.optim as optim
from pytorch_points.misc import logger
from pytorch_points.network.geo_operations import mean_value_coordinates_3D, edge_vertex_indices
from pytorch_points.utils.pc_utils import load, save_ply, save_pts, center_bounding_box
from pytorch_points.utils.geometry_utils import read_trimesh, write_trimesh, build_gemm, Mesh, get_edge_points, generatePolygon
from pytorch_points.utils.pytorch_utils import weights_init, check_values, save_network, load_network, save_grad, saved_variables, \
    clamp_gradient_norm, linear_loss_weight, tolerating_collate, clamp_gradient, fix_network_parameters
import losses
import networks
from common import loadInitCage, build_dataset, crisscross_input, log_outputs, deform_with_MVC
from model import DGCNN, PointNetCls, feature_transform_reguliarzer, \
    PointNet2ClsSsg, PointConvDensityClsSsg
from adv_utils import LogitsAdvLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import math


def test(net=None, save_subdir="test"):
    opt.phase = "test"
    dataset = build_dataset(opt)

    if opt.dim == 3:
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor([(x, y) for x, y in init_cage_V], dtype=torch.float).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        init_cage_Fs = [torch.arange(opt.cage_deg, dtype=torch.int64).view(1,1,-1).cuda()]

    if net is None:
        # network
        net = networks.NetworkFull(opt, dim=opt.dim, bottleneck_size=opt.bottleneck_size,
                               template_vertices=cage_V_t, template_faces=init_cage_Fs[-1],
                               ).cuda()
        net.eval()
        load_cage_checkpoint(net, opt.ckpt)
    else:
        net.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                                             collate_fn=tolerating_collate,
                                             num_workers=0,
                                             worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    test_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)

    with open(os.path.join(test_output_dir, "eval.txt"), "w") as f:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                data = dataset.uncollate(data)

                # Blending logic
                if opt.blend_style:
                    num_alpha = 4
                    blend_alpha = torch.linspace(0, 1, steps=num_alpha, dtype=torch.float32).cuda().reshape(num_alpha, 1)
                    data["source_shape"] = data["source_shape"].expand(num_alpha, -1, -1).contiguous()
                    data["target_shape"] = data["target_shape"].expand(num_alpha, -1, -1).contiguous()
                else:
                    blend_alpha = 1.0

                data["alpha"] = blend_alpha

                # Running the network
                source_shape_t = data["source_shape"].transpose(1,2).contiguous().detach()
                target_shape_t = data["target_shape"].transpose(1,2).contiguous().detach()

                outputs = net(source_shape_t, target_shape_t, blend_alpha)
                deformed = outputs["deformed"]

                s_filename = os.path.splitext(data["source_file"][0])[0]
                t_filename = os.path.splitext(data["target_file"][0])[0]

                log_str = "{}/{} {}-{} ".format(i, len(dataloader), s_filename, t_filename)
                print(log_str)
                f.write(log_str+"\n")

                # Save outputs (for visual inspection or further processing)
                for b in range(deformed.shape[0]):
                    # save to "pts" for rendering
                    save_pts(os.path.join(opt.log_dir, save_subdir,"{}-{}-Sa.pts".format(s_filename,t_filename)), data["source_shape"][b].detach().cpu())
                    save_pts(os.path.join(opt.log_dir, save_subdir,"{}-{}-Sab-{}.pts".format(s_filename,t_filename, b)), deformed[0].detach().cpu())

                    save_pts(os.path.join(opt.log_dir, save_subdir,"{}-{}-Sb.pts".format(s_filename,t_filename)), data["target_shape"][0].detach().cpu())

                    outputs["new_cage"][b] = center_bounding_box(outputs["new_cage"][b])[0]
                    pymesh.save_mesh_raw(
                        os.path.join(opt.log_dir, save_subdir, "{}-{}-cage1-{}.ply".format(s_filename, t_filename, b)),
                        outputs["cage"][b].detach().cpu(), outputs["cage_face"][0].detach().cpu(), binary=True)
                    pymesh.save_mesh_raw(
                        os.path.join(opt.log_dir, save_subdir, "{}-{}-cage2-{}.ply".format(s_filename, t_filename, b)),
                        outputs["new_cage"][b].detach().cpu(), outputs["cage_face"][0].detach().cpu(), binary=True)




import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_point_cloud(pc, output_dir, file_name, point_size=1):
    """ 포인트 클라우드를 시각화하여 이미지로 저장하는 함수 """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=point_size, c='b', marker='o')
    
    # 축 제한 설정 (-1~1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])


    # 이미지 저장
    output_path = os.path.join(output_dir, f"{file_name}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def train():
    dataset = build_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                             collate_fn=tolerating_collate, num_workers=2, 
                                             worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    if opt.dim == 3:
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        print(opt.template)
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
        cage_edge_points_list = []
        cage_edges_list = []
        for F in init_cage_Fs:
            mesh = Mesh(vertices=init_cage_V[0], faces=F[0])
            build_gemm(mesh, F[0])
            cage_edge_points = torch.from_numpy(get_edge_points(mesh)).cuda()
            cage_edge_points_list.append(cage_edge_points)
            cage_edges_list = [edge_vertex_indices(F[0])]
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor([(x, y) for x, y in init_cage_V], dtype=torch.float).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
        init_cage_Fs = [torch.arange(opt.cage_deg, dtype=torch.int64).view(1, 1, -1).cuda()]

    # NetworkFull 초기화
    net = networks.NetworkFull(opt, dim=opt.dim, bottleneck_size=opt.bottleneck_size, template_vertices=cage_V_t, template_faces=init_cage_Fs[-1]).cuda()

    net.apply(weights_init)
    if opt.ckpt:
        load_cage_checkpoint(net, opt.ckpt)

    all_losses = losses.AllLosses(opt)
    optimizer = torch.optim.Adam([{"params": net.encoder.parameters()},
                                  {"params": net.nd_decoder.parameters()},
                                  {"params": net.merger.parameters()}], lr=opt.lr)

    if opt.full_net:
        optimizer.add_param_group({'params': net.nc_decoder.parameters(), 'lr': 0.1*opt.lr})
    if opt.optimize_template:
        optimizer.add_param_group({'params': net.template_vertices, 'lr': opt.lr})

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs * 0.4), gamma=0.1, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1, last_epoch=-1)

    net.train()
    start_epoch = 0
    t = 0
    steps_C = 20
    steps_D = 20

    source_shape_saved = False
    all_source_shapes = []

    # npz 저장을 위한 새로운 폴더 생성
    visualization_dir = os.path.join(opt.log_dir, 'visualization')
    os.makedirs(visualization_dir, exist_ok=True)

    os.makedirs(opt.log_dir, exist_ok=True)
    shutil.copy2(__file__, opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "networks.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "losses.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "datasets.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "common.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "option.py"), opt.log_dir)

    log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    log_file.write(str(net) + "\n")

    log_interval = max(len(dataloader) // 5, 50)
    save_interval = max(opt.nepochs // 10, 1)
    running_avg_loss = -1

    with torch.autograd.detect_anomaly():
        if opt.epoch:
            start_epoch = opt.epoch % opt.nepochs
            t += start_epoch * len(dataloader)

        for epoch in range(start_epoch, opt.nepochs):
            for t_epoch, data in enumerate(dataloader):
                warming_up = epoch < opt.warmup_epochs
                optimize_C = (t % (steps_C + steps_D)) > steps_D

                data = dataset.uncollate(data)
                data = crisscross_input(data)

                source_shape, target_shape = data["source_shape"], data["target_shape"]

                # source_shape 데이터 저장
                if not all_source_shapes:
                    all_source_shapes.append(source_shape[:8].cpu().numpy())

                if opt.blend_style:
                    blend_alpha = torch.rand((source_shape.shape[0], 1), dtype=torch.float32).to(device=source_shape.device)
                else:
                    blend_alpha = 1.0
                data["alpha"] = blend_alpha

                optimizer.zero_grad()
                source_shape_t = source_shape.transpose(1, 2)
                target_shape_t = target_shape.transpose(1, 2)
                outputs = net(source_shape_t, target_shape_t, data["alpha"])

                current_loss = all_losses(data, outputs, epoch)
                loss_sum = torch.sum(torch.stack([v for v in current_loss.values()], dim=0))

                if running_avg_loss < 0:
                    running_avg_loss = loss_sum
                else:
                    running_avg_loss = running_avg_loss + (loss_sum.item() - running_avg_loss) / (t + 1)

                if (t % log_interval == 0) or (loss_sum > 5 * running_avg_loss):
                    log_str = f"warming up {warming_up} e {epoch:03d} t {t:05d}: " + ", ".join([f"{k} {v.mean().item():.3g}" for k, v in current_loss.items()])
                    print(log_str)
                    log_file.write(log_str + "\n")
                    log_outputs(opt, t, outputs, data)

                if loss_sum > 100 * running_avg_loss:
                    logger.info(f"loss ({loss_sum}) > 5*running_average_loss ({5*running_avg_loss}). Skip without update.")
                    torch.cuda.empty_cache()
                    continue

                loss_sum.backward()
                clamp_gradient(net, 0.1)
                optimizer.step()

                if (t + 1) % 500 == 0:
                    save_network(net, opt.log_dir, network_label="net", epoch_label="latest")

                t += 1

            if (epoch + 1) % save_interval == 0:
                save_network(net, opt.log_dir, network_label="net", epoch_label=epoch)

            scheduler.step()
            if opt.eval:
                try:
                    test(net=net, save_subdir=f"epoch_{epoch}")
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)
                    logger.warn("Failed to run test", str(e))

    # source_shape 데이터 한번만 저장 및 시각화
    if not source_shape_saved and len(all_source_shapes) > 0:
        # 시각화
        for i, shape in enumerate(all_source_shapes[0]):
            visualize_point_cloud(shape, visualization_dir, f"source_shape_{i}", point_size=1)

        # npz 저장
        np.savez(os.path.join(opt.log_dir, 'source_shape_data.npz'), test_pc=np.concatenate(all_source_shapes, axis=0))
        source_shape_saved = True

    log_file.close()
    save_network(net, opt.log_dir, network_label="net", epoch_label="final")
    test(net=net)


#=========================================================================================================================================================================

# 경로 리스트를 입력받아 각 파일의 라벨을 반환하는 함수




def adv_train():
    dataset = build_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                             collate_fn=tolerating_collate,
                                             num_workers=2, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
    
    if opt.dim == 3:
        # cage (1,N,3)
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        cage_edge_points_list = []
        cage_edges_list = []
        for F in init_cage_Fs:
            mesh = Mesh(vertices=init_cage_V[0], faces=F[0])
            build_gemm(mesh, F[0])
            cage_edge_points = torch.from_numpy(get_edge_points(mesh)).cuda()
            cage_edge_points_list.append(cage_edge_points)
            cage_edges_list = [edge_vertex_indices(F[0])]
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor([(x, y) for x, y in init_cage_V], dtype=torch.float).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        init_cage_Fs = [torch.arange(opt.cage_deg, dtype=torch.int64).view(1,1,-1).cuda()]

    # network
    net = networks.NetworkFull(opt, dim=opt.dim, bottleneck_size=opt.bottleneck_size,
                               template_vertices=cage_V_t, template_faces=init_cage_Fs[-1],
                               ).cuda()
    
    model = load_classifier(opt)
    net.apply(weights_init)
    load_cage_checkpoint(net, opt.ckpt)

    # net.nd_decoder 외 모든 네트워크의 파라미터를 업데이트하지 않도록 설정
    for name, param in net.named_parameters():
        if "nd_decoder" not in name:
            param.requires_grad = False

    all_losses = losses.AllLosses(opt)
    
    # optimizer: net.nd_decoder 파라미터만 업데이트
    optimizer = torch.optim.Adam([
        {"params": net.nd_decoder.parameters()}], lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(int(opt.nepochs*0.4), 1), gamma=0.1, last_epoch=-1)


    # train
    net.eval()
    net.nd_decoder.train()
    start_epoch = 0
    t = 0

    steps_C = 20
    steps_D = 20

    # train
    os.makedirs(opt.log_dir, exist_ok=True)
    shutil.copy2(__file__, opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "networks.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "losses.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "datasets.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "common.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "option.py"), opt.log_dir)

    log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    log_file.write(str(net)+"\n")

    log_interval = max(len(dataloader)//5, 50)
    save_interval = max(opt.nepochs//10, 1)
    running_avg_loss = -1

    with torch.autograd.detect_anomaly():
        if opt.epoch:
            start_epoch = opt.epoch % opt.nepochs
            t += start_epoch*len(dataloader)

        for epoch in range(start_epoch, opt.nepochs):
            for t_epoch, data in enumerate(dataloader):
                warming_up = epoch < opt.warmup_epochs
                progress = t_epoch/len(dataloader)+epoch
                optimize_C = (t % (steps_C+steps_D)) > steps_D

                ############# get data ###########
                data = dataset.uncollate(data)
                #data = crisscross_input(data)
                if opt.dim == 3:
                    data["cage_edge_points"] = cage_edge_points_list[-1]
                    data["cage_edges"] = cage_edges_list[-1]
                source_shape, target_shape = data["source_shape"], data["target_shape"]

                source_label = torch.tensor(classification_labels(data, "source", opt), dtype=torch.long).cuda()
                target_label = torch.tensor(classification_labels(data, "target", opt), dtype=torch.long).cuda()

                ############# blending ############
                if opt.blend_style:
                    blend_alpha = torch.rand((source_shape.shape[0], 1), dtype=torch.float32).to(device=source_shape.device)
                else:
                    blend_alpha = 1.0
                data["alpha"] = blend_alpha

                ############# run network ###########
                optimizer.zero_grad()
                source_shape_t = source_shape.transpose(1,2).contiguous()
                target_shape_t = target_shape.transpose(1,2).contiguous()

                outputs = net(source_shape_t, target_shape_t, data["alpha"])

                logits = model(outputs["deformed"].transpose(1,2))  # [B, num_classes]

                if isinstance(logits, tuple):  # PointNet
                    logits = logits[0]

                preds = torch.argmax(logits, dim=-1)

                ############# get losses ###########
                current_loss = all_losses(data, outputs, progress)
                adv_loss = adv_func(logits, source_label).mean()
                current_loss["adv_loss"] = adv_loss * opt.adv_weight
                current_loss.pop("MSE", None)
                current_loss.pop("SCD", None) 
                
                loss_sum = torch.sum(torch.stack([v for v in current_loss.values()], dim=0))
                if running_avg_loss < 0:
                    running_avg_loss = loss_sum
                else:
                    running_avg_loss = running_avg_loss + (loss_sum.item() - running_avg_loss)/(t+1)

                if (t % log_interval == 0) or (loss_sum > 5 * running_avg_loss):
                    # 공격 성공 여부를 계산
                    attack_success_count = (preds != source_label).sum().item()  # preds와 source_label이 다른 경우의 수
                    log_str = "warming up {} e {:03d} t {:05d}: {} | Attack Success Count: {}".format(
                        warming_up, epoch, t,
                        ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()]),
                        attack_success_count
                    )
                    print(log_str)

                if loss_sum > 100*running_avg_loss:
                    logger.info("loss ({}) > 5*running_average_loss ({}). Skip without update.".format(loss_sum, 5*running_avg_loss))
                    torch.cuda.empty_cache()
                    continue

                loss_sum.backward()
                clamp_gradient(net, 0.1)
                optimizer.step()

                if (t + 1) % 500 == 0:
                    save_network(net, opt.log_dir, network_label="net", epoch_label="latest")

                t += 1

            if (epoch + 1) % save_interval == 0:
                save_network(net, opt.log_dir, network_label="net", epoch_label=epoch)

            scheduler.step()
            if opt.eval:
                try:
                    adv_test(net=net, save_subdir="epoch_{}".format(epoch))
                    opt.phase = "adv_train"
                    net.eval()
                    net.nd_decoder.train()
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)
                    logger.warn("Failed to run test", str(e))

    log_file.close()
    save_network(net, opt.log_dir, network_label="net", epoch_label="final")
    if opt.eval:
        adv_test(net=net)

def adv_test(net=None, save_subdir="test"):
    if opt.blend_style:
        raise ValueError("Disable --blend_style for unambiguous ASR evaluation")
    opt.phase = "test"
    dataset = build_dataset(opt)

    if opt.dim == 3:
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor(
            [(x, y) for x, y in init_cage_V], dtype=torch.float
        ).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1, 2).detach().cuda()
        init_cage_Fs = [
            torch.arange(opt.cage_deg, dtype=torch.int64).view(1, 1, -1).cuda()
        ]

    if net is None:
        # Network
        net = networks.NetworkFull(
            opt,
            dim=opt.dim,
            bottleneck_size=opt.bottleneck_size,
            template_vertices=cage_V_t,
            template_faces=init_cage_Fs[-1],
        ).cuda()
        net.eval()
        load_cage_checkpoint(net, opt.ckpt)
    else:
        net.eval()

    model = load_classifier(opt)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=tolerating_collate,
        num_workers=0,
        worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id),
    )

    test_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)

    records = []
    with open(os.path.join(test_output_dir, "eval.txt"), "w") as f:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                data = dataset.uncollate(data)

                # Blending logic
                if opt.blend_style:
                    num_alpha = 4
                    blend_alpha = (
                        torch.linspace(0, 1, steps=num_alpha, dtype=torch.float32)
                        .cuda()
                        .reshape(num_alpha, 1)
                    )
                    data["source_shape"] = (
                        data["source_shape"]
                        .expand(num_alpha, -1, -1)
                        .contiguous()
                    )
                    data["target_shape"] = (
                        data["target_shape"]
                        .expand(num_alpha, -1, -1)
                        .contiguous()
                    )
                else:
                    blend_alpha = 1.0

                data["alpha"] = blend_alpha

                # Obtain labels
                source_label = torch.tensor(
                    classification_labels(data, "source", opt),
                    dtype=torch.long,
                ).cuda()
                target_label = torch.tensor(
                    classification_labels(data, "target", opt),
                    dtype=torch.long,
                ).cuda()

                # Running the network
                source_shape_t = (
                    data["source_shape"].transpose(1, 2).contiguous().detach()
                )
                target_shape_t = (
                    data["target_shape"].transpose(1, 2).contiguous().detach()
                )

                outputs = net(source_shape_t, target_shape_t, blend_alpha)
                deformed = outputs["deformed"]

                # Get predictions from classifier
                logits = model(deformed.transpose(1, 2))
                if isinstance(logits, tuple):  # For models like PointNet
                    logits = logits[0]
                preds = torch.argmax(logits, dim=-1)
                clean_logits = model(source_shape_t)
                if isinstance(clean_logits, tuple):
                    clean_logits = clean_logits[0]
                records.append({"source": data["source_file"][0],
                                "target": data["target_file"][0],
                                "label": source_label[0].item(),
                                "clean_prediction": clean_logits.argmax(-1)[0].item(),
                                "adversarial_prediction": preds[0].item()})


                s_filename = os.path.splitext(data["source_file"][0])[0]
                t_filename = os.path.splitext(data["target_file"][0])[0]

                log_str = "{}/{} {}-{} Predicted: {} True: {}".format(
                    i,
                    len(dataloader),
                    s_filename,
                    t_filename,
                    preds[0].item(),
                    source_label[0].item(),
                )
                print(log_str)
                f.write(log_str + "\n")

                # Save outputs (for visual inspection or further processing)
                for b in range(deformed.shape[0]):
                    # Save to "pts" for rendering, including predicted class in filename
                    save_pts(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-Sab-{}-pred{}.pts".format(
                                s_filename, t_filename, b, preds[b].item()
                            ),
                        ),
                        deformed[b].detach().cpu(),
                    )

                    # Optionally, save source and target shapes with true labels
                    save_pts(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-Sa-true{}.pts".format(
                                s_filename, t_filename, source_label[b].item()
                            ),
                        ),
                        data["source_shape"][b].detach().cpu(),
                    )
                    save_pts(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-Sb-true{}.pts".format(
                                s_filename, t_filename, target_label[b].item()
                            ),
                        ),
                        data["target_shape"][b].detach().cpu(),
                    )

                    # Save cages if necessary
                    pymesh.save_mesh_raw(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-cage1-{}.ply".format(s_filename, t_filename, b),
                        ),
                        outputs["cage"][b].detach().cpu(),
                        outputs["cage_face"][0].detach().cpu(),
                        binary=True,
                    )
                    pymesh.save_mesh_raw(
                        os.path.join(
                            opt.log_dir,
                            save_subdir,
                            "{}-{}-cage2-{}.ply".format(s_filename, t_filename, b),
                        ),
                        outputs["new_cage"][b].detach().cpu(),
                        outputs["cage_face"][0].detach().cpu(),
                        binary=True,
                    )





    with open(os.path.join(test_output_dir, "predictions.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(test_output_dir, "metrics.json"), "w") as f:
        json.dump(attack_metrics(records), f, indent=2, allow_nan=False)


def main():
    from option import BaseOptions
    import datetime
    global opt, adv_func
    parser = BaseOptions()
    opt = parser.parse()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("This implementation requires Linux, CUDA and compiled pytorch_points extensions")
    if opt.phase in ("adv_train", "adv_test") and not opt.ckpt:
        parser.parser.error("--ckpt is required for adversarial training/evaluation")
    if opt.phase in ("adv_train", "adv_test") and (opt.dataset not in ("SHAPENET", "MN40", "SCANOBJECTNN") or opt.data_cat == "*"):
        parser.parser.error("Select SHAPENET/MN40/SCANOBJECTNN and one --data_cat category")
    if opt.phase in ("train", "adv_train"):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        opt.log_dir = os.path.join(opt.log_dir, "cage-{}-{}".format(stamp, opt.name or opt.data_cat))
    else:
        # Preserve the caller's output directory rather than writing beside checkpoints.
        opt.log_dir = os.path.abspath(opt.log_dir)
    os.makedirs(opt.log_dir, exist_ok=True)
    with open(os.path.join(opt.log_dir, "run_config.json"), "w") as f:
        json.dump(vars(opt), f, indent=2)
    adv_func = LogitsAdvLoss(kappa=opt.margin)
    {"train": train, "adv_train": adv_train, "test": test,
     "adv_test": lambda: adv_test(save_subdir=opt.subdir)}[opt.phase]()


if __name__ == "__main__":
    main()
