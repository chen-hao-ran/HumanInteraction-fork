'''
 @FileName    : process.py
 @EditTime    : 2022-09-27 16:18:51
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def merge_gt(data):
    batch_size, frame_length, agent_num = data['pose'].shape[:3]

    data['data_shape'] = data['pose'].shape[:3]
    data['has_3d'] = data['has_3d'].reshape(batch_size*frame_length*agent_num,1)
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*frame_length*agent_num,1)
    data['verts'] = data['verts'].reshape(batch_size*frame_length*agent_num, 6890, 3)
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*frame_length*agent_num, -1, 4)
    data['pose'] = data['pose'].reshape(batch_size*frame_length*agent_num, 72)
    data['betas'] = data['betas'].reshape(batch_size*frame_length*agent_num, 10)
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*frame_length*agent_num, 3)
    data['x'] = data['x'].reshape(batch_size*frame_length*agent_num, -1)

    imgname = (np.array(data['imgname']).T).reshape(batch_size*frame_length,)
    data['imgname'] = imgname.tolist()

    return data

def extract_valid(data):
    batch_size, frame_length, agent_num = data['keypoints'].shape[:3]

    data['data_shape'] = data['keypoints'].shape[:3]
    data['center'] = data['center'].reshape(batch_size*frame_length*agent_num, -1)
    data['scale'] = data['scale'].reshape(batch_size*frame_length*agent_num,)
    data['img_h'] = data['img_h'].reshape(batch_size*frame_length*agent_num,)
    data['img_w'] = data['img_w'].reshape(batch_size*frame_length*agent_num,)
    data['focal_length'] = data['focal_length'].reshape(batch_size*frame_length*agent_num,)
    data['valid'] = data['valid'].reshape(batch_size*frame_length*agent_num,)

    data['has_3d'] = data['has_3d'].reshape(batch_size*frame_length*agent_num,1)
    data['has_smpl'] = data['has_smpl'].reshape(batch_size*frame_length*agent_num,1)
    data['verts'] = data['verts'].reshape(batch_size*frame_length*agent_num, -1, 3)
    data['gt_joints'] = data['gt_joints'].reshape(batch_size*frame_length*agent_num, -1, 4)
    data['pose'] = data['pose'].reshape(batch_size*frame_length*agent_num, 72)
    data['betas'] = data['betas'].reshape(batch_size*frame_length*agent_num, 10)
    data['keypoints'] = data['keypoints'].reshape(batch_size*frame_length*agent_num, 26, 3)
    data['gt_cam_t'] = data['gt_cam_t'].reshape(batch_size*frame_length*agent_num, 3)

    imgname = (np.array(data['imgname']).T).reshape(batch_size*frame_length,)
    data['imgname'] = imgname.tolist()

    return data

def extract_valid_demo(data):
    batch_size, agent_num, _, _, _ = data['norm_img'].shape
    valid = data['valid'].reshape(-1,)

    data['center'] = data['center'].reshape(batch_size*agent_num, -1)[valid == 1]
    data['scale'] = data['scale'].reshape(batch_size*agent_num,)[valid == 1]
    data['img_h'] = data['img_h'].reshape(batch_size*agent_num,)[valid == 1]
    data['img_w'] = data['img_w'].reshape(batch_size*agent_num,)[valid == 1]
    data['focal_length'] = data['focal_length'].reshape(batch_size*agent_num,)[valid == 1]

    # imgname = (np.array(data['imgname']).T).reshape(batch_size*agent_num,)[valid.detach().cpu().numpy() == 1]
    # data['imgname'] = imgname.tolist()

    return data

def to_device(data, device):
    imnames = {'imgname':data['imgname']} 
    data = {k:v.to(device).float() for k, v in data.items() if k not in ['imgname']}
    data = {**imnames, **data}

    return data

def reconstruction_train(model, loss_func, train_loader, epoch, num_epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    if model.scheduler is not None:
        model.scheduler.step()

    train_loss = 0.
    for i, data in enumerate(train_loader):
        batchsize = data['keypoints'].shape[0]
        data = to_device(data, device)
        data = extract_valid(data)

        # forward
        pose = data["pose"].view(-1, 2, 72)[:, 0]
        pred = model.model(pose, data)

        # calculate loss
        pose_loss = torch.sum((pred["pred_pose"] - data["pose"].view(-1, 2, 72)[:, 1]) ** 2)
        criterion = nn.BCELoss(reduction='none')
        segmentation_loss = criterion(pred["pred_segmentation"], data["segmentation"].view(-1, 150))
        segmentation_loss = torch.sum(torch.sum(segmentation_loss, dim=1)) / batchsize
        signature_loss = criterion(pred["pred_signature"], data["signature"].view(-1, 5625))
        signature_loss = torch.sum(torch.sum(signature_loss, dim=1)) / batchsize

        loss = 0.01 * pose_loss + 0.01 * segmentation_loss + 0.01 * signature_loss

        # iou
        pred_segmentation_binary = torch.threshold(pred["pred_segmentation"], 0.75, 0).to(torch.int)
        pred_signature_binary = torch.threshold(pred["pred_signature"], 0.75, 0).to(torch.int)
        segmentation_intersection = torch.sum(pred_segmentation_binary & data["segmentation"].reshape(-1, 150).to(torch.int))
        segmentation_union = torch.sum(pred_segmentation_binary | data["segmentation"].view(-1, 150).to(torch.int))
        signature_intersection = torch.sum(pred_signature_binary & data["signature"].view(-1, 5625).to(torch.int))
        signature_union = torch.sum(pred_signature_binary | data["signature"].reshape(-1, 5625).to(torch.int))
        segmentation_iou = segmentation_intersection / segmentation_union
        signature_iou = signature_intersection / signature_union
        print(torch.max(pred["pred_segmentation"]))
        print('[%d/%d] segmentation_iou: %.4f' % (i + 1, len_data, segmentation_iou))
        print('[%d/%d] signature_iou: %.4f' % (i + 1, len_data, signature_iou))

        # mpjpe
        pred_joints = pred["pred_3dkp"]
        gt_joints = data["gt_joints"].view(-1, 2, 26, 4)[:, 1]

        halpe2lsp = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 18, 17]
        conf = gt_joints[:, halpe2lsp, -1]

        pred_joints = pred_joints[:, halpe2lsp]
        gt_joints = gt_joints[:, halpe2lsp, :3]

        pred_joints = align_by_pelvis(pred_joints, format='lsp')
        gt_joints = align_by_pelvis(gt_joints, format='lsp')

        # gui.vis_skeleton(pred_joints.detach().cpu().numpy(), gt_joints.detach().cpu().numpy(), format='lsp')

        diff = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=[2]) * conf)
        diff = torch.mean(diff, dim=[1])
        diff = torch.mean(diff) * 1000
        print('[%d/%d] mpjpe: %.4f' % (i + 1, len_data, diff))

        # pa-mpjpe
        # pred_joints = batch_compute_similarity_transform(pred_joints, gt_joints)

        # diff = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=[2]) * conf)
        # diff = torch.mean(diff, dim=[1])
        # diff = torch.mean(diff) * 1000
        # print('[%d/%d] pa-mpjpe: %.4f' % (i + 1, len_data, diff))

        # backward
        model.optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(parameters=model.model.parameters(), max_norm=100, norm_type=2)

        # optimize
        model.optimizer.step()
        if model.scheduler is not None:
            model.scheduler.batch_step()

        loss_batch = loss.detach() #/ batchsize
        # print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), cur_loss_dict)
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' % (epoch, num_epoch, i, len_data, loss_batch), loss_batch)
        train_loss += loss_batch

    return train_loss/len_data

def reconstruction_test(model, loss_func, loader, epoch, device=torch.device('cpu')):

    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['keypoints'].shape[0]
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pose = data["pose"].view(-1, 2, 72)[:, 0]
            pred = model.model(pose, data)

            # calculate loss
            pred_joints = pred["pred_3dkp"]
            gt_joints = data["gt_joints"].view(-1, 2, 26, 4)[:, 1]

            halpe2lsp = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, 18, 17]
            conf = gt_joints[:, halpe2lsp, -1]

            pred_joints = pred_joints[:, halpe2lsp]
            gt_joints = gt_joints[:, halpe2lsp, :3]

            pred_joints = align_by_pelvis(pred_joints, format='lsp')
            gt_joints = align_by_pelvis(gt_joints, format='lsp')

            # gui.vis_skeleton(pred_joints.detach().cpu().numpy(), gt_joints.detach().cpu().numpy(), format='lsp')

            diff = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=[2]) * conf)
            diff = torch.mean(diff, dim=[1])
            loss_mpjpe = torch.mean(diff) * 1000

            pred_joints = batch_compute_similarity_transform(pred_joints, gt_joints)

            diff = torch.sqrt(torch.sum((pred_joints - gt_joints) ** 2, dim=[2]) * conf)
            diff = torch.mean(diff, dim=[1])
            loss_pa_mpjpe = torch.mean(diff) * 1000

            # 可视化
            if False:
                pred = pred.view(-1, 1, 72).repeat(1, 2, 1)
                pred[:, 0] = data["pose"].view(-1, 2, 72)[:, 0]
                pred = pred.view(-1, 72)
                shape = data["betas"]
                trans = data["gt_cam_t"]
                pred_verts, pred_joints = model.model.smpl(shape, pred, trans, halpe=True)
                pred_verts = pred_verts.view(-1, 2, 6890, 3)
                pred_verts = torch.cat((pred_verts[:, 0], pred_verts[:, 1]), dim=1)
                pred_verts = pred_verts.detach().cpu().numpy().astype(np.float32)

                import os
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                for i, verts in enumerate(pred_verts):
                    output_path = os.path.join(output_dir, f"{i:06d}.txt")
                    np.savetxt(output_path, verts)

            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['img_w'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                model.save_params(results, i, batchsize)


            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(single_person=data['single_person'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                if 'MPJPE_instance' in cur_loss_dict.keys():
                    results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))
                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    model.save_results(results, i, batchsize)

            loss_batch_mpjpe = loss_mpjpe.mean().detach() #/ batchsize
            loss_batch_pa_mpjpe = loss_pa_mpjpe.mean().detach()
            print('batch: %d/%d, mpjpe: %.6f pa-mpjpe: %.6f' %(i, len(loader), loss_batch_mpjpe, loss_batch_pa_mpjpe), loss_batch_mpjpe)
            loss_all += loss_batch_mpjpe + loss_pa_mpjpe
        loss_all = loss_all / len(loader)
        return loss_all

def reconstruction_eval(model, loader, loss_func, device=torch.device('cpu')):

    print('-' * 10 + 'model eval' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    output = {'pose':{}, 'shape':{}, 'trans':{}}
    gt = {'pose':{}, 'shape':{}, 'trans':{}, 'gender':{}, 'valid':{}}
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            # if i > 1:
            #     break
            batchsize = data['keypoints'].shape[0]
            seq_id = data['seq_id']
            frame_id = torch.cat(data['frame_id']).reshape(-1, batchsize)
            frame_id = frame_id.detach().cpu().numpy().T

            batch_size, frame_length, agent_num = data['keypoints'].shape[:3]

            del data['seq_id']
            del data['frame_id']
            data = to_device(data, device)
            data = extract_valid(data)

            # forward
            pose = data["pose"].view(-1, 2, 72)[:, 0]
            pred = model.model(pose, data)

            # pred_pose = pred['pred_pose'].reshape(batch_size, frame_length, agent_num, -1)
            # pred_shape = pred['pred_shape'].reshape(batch_size, frame_length, agent_num, -1)
            # pred_trans = pred['pred_cam_t'].reshape(batch_size, frame_length, agent_num, -1)
            #
            # pred_pose = pred_pose.detach().cpu().numpy()
            # pred_shape = pred_shape.detach().cpu().numpy()
            # pred_trans = pred_trans.detach().cpu().numpy()
            #
            # gt_pose = data['pose'].reshape(batch_size, frame_length, agent_num, -1)
            # gt_shape = data['betas'].reshape(batch_size, frame_length, agent_num, -1)
            # gt_trans = data['gt_cam_t'].reshape(batch_size, frame_length, agent_num, -1)
            # gt_gender = data['gender'].reshape(batch_size, frame_length, agent_num)
            # valid = data['valid'].reshape(batch_size, frame_length, agent_num)
            #
            # gt_pose = gt_pose.detach().cpu().numpy()
            # gt_shape = gt_shape.detach().cpu().numpy()
            # gt_trans = gt_trans.detach().cpu().numpy()
            # gt_gender = gt_gender.detach().cpu().numpy()
            # valid = valid.detach().cpu().numpy()
            #
            # for batch in range(batchsize):
            #     s_id = str(int(seq_id[batch]))
            #     for f in range(frame_length):
            #
            #         if s_id not in output['pose'].keys():
            #             output['pose'][s_id] = [pred_pose[batch][f]]
            #             output['shape'][s_id] = [pred_shape[batch][f]]
            #             output['trans'][s_id] = [pred_trans[batch][f]]
            #
            #             gt['pose'][s_id] = [gt_pose[batch][f]]
            #             gt['shape'][s_id] = [gt_shape[batch][f]]
            #             gt['trans'][s_id] = [gt_trans[batch][f]]
            #             gt['gender'][s_id] = [gt_gender[batch][f]]
            #             gt['valid'][s_id] = [valid[batch][f]]
            #         else:
            #             output['pose'][s_id].append(pred_pose[batch][f])
            #             output['shape'][s_id].append(pred_shape[batch][f])
            #             output['trans'][s_id].append(pred_trans[batch][f])
            #
            #             gt['pose'][s_id].append(gt_pose[batch][f])
            #             gt['shape'][s_id].append(gt_shape[batch][f])
            #             gt['trans'][s_id].append(gt_trans[batch][f])
            #             gt['gender'][s_id].append(gt_gender[batch][f])
            #             gt['valid'][s_id].append(valid[batch][f])
            
            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_pose=pred['pred_pose'].detach().cpu().numpy().astype(np.float32))
                results.update(pred_shape=pred['pred_shape'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_pose=data['pose'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_shape=data['betas'].detach().cpu().numpy().astype(np.float32))
                results.update(img_h=data['img_h'].detach().cpu().numpy().astype(np.float32))
                results.update(img_w=data['img_w'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                model.save_params(results, i, batchsize)


            if False: #loss.max() > 100:
                results = {}
                results.update(imgs=data['imgname'])
                results.update(pred_trans=pred['pred_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(gt_trans=data['gt_cam_t'].detach().cpu().numpy().astype(np.float32))
                results.update(focal_length=data['focal_length'].detach().cpu().numpy().astype(np.float32))
                results.update(MPJPE=loss.detach().cpu().numpy().astype(np.float32))
                if 'pred_verts' not in pred.keys():
                    results.update(pred_joints=pred['pred_joints'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_joints=data['gt_joints'].detach().cpu().numpy().astype(np.float32))
                    model.save_joint_results(results, i, batchsize)
                else:
                    results.update(pred_verts=pred['pred_verts'].detach().cpu().numpy().astype(np.float32))
                    results.update(gt_verts=data['verts'].detach().cpu().numpy().astype(np.float32))
                    model.save_results(results, i, batchsize)

        return output, gt

def align_by_pelvis(joints, format='lsp'):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    if format == 'lsp':
        left_id = 3
        right_id = 2

        pelvis = (joints[:,left_id, :] + joints[:,right_id, :]) / 2.
    elif format in ['smpl', 'h36m']:
        pelvis_id = 0
        pelvis = joints[pelvis_id, :]
    elif format in ['mpi']:
        pelvis_id = 14
        pelvis = joints[pelvis_id, :]

    return joints - pelvis[:,None,:].repeat(1, 14, 1)

def batch_compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    t1 = U.bmm(V.permute(0,2,1))
    t2 = torch.det(t1)
    Z[:,-1, -1] = Z[:,-1, -1] * torch.sign(t2)
    # Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat

