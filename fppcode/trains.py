import torch
import numpy as np
from torch import nn
import torch.nn.functional as F 
from sklearn.metrics import roc_auc_score

def test_2cls_celoss_attn(loader, gnn_model, args):
    gnn_model.eval()
    roc_auc = []

    emb = []
    contri = []
    masklist = []
    smileslist = []
    ylist = []

    for data in loader:
        data = data.to(args['device'])
        out, embed, contributes, mask, pool_loss  = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        score = F.softmax(out, dim=-1)
        pre_label = score[:, 1]

        emb.append(embed.cpu())
        contri.append(contributes)
        masklist.append(mask)
        ylist.append(data.y)
        smileslist.append(data.smiles)

        try:
            roc_auc.append(roc_auc_score(data.y.cpu().numpy(), pre_label.cpu().numpy(), average='macro'))
        except:
            continue
    return np.mean(np.array(roc_auc)), emb, ylist, contri, masklist, smileslist 
@torch.no_grad()
def test_muti2cls_bceloss_attn(loader, gnn_model, args):
    gnn_model.eval()

    emb = []
    contri = []
    masklist = []
    smileslist = []
    ylist = []

    pred_list = []
    label_list = []
    roc_auc_mi = []
    roc_auc_ma = []
    total_loss = total_examples = 0
    for i,data in enumerate(loader):
        roc_auc_temp_mi = []
        roc_auc_temp_ma = []
        data = data.to(args['device'])
        out, embed, contributes, mask, pool_loss = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        loss = nn.BCEWithLogitsLoss()
        nn_loss = loss(out, data.y.squeeze().to(torch.float)).detach().cpu()
        total_loss += float(nn_loss) * data.num_graphs
        total_examples += data.num_graphs

        emb.append(embed.cpu())
        contri.append(contributes)
        masklist.append(mask)
        ylist.append(data.y)
        smileslist.append(data.smiles)

        score = F.sigmoid(out)
        pre_label = score
        true_label = data.y.squeeze().to(torch.long)
        pred_list.append(pre_label)
        label_list.append(true_label)


        for i in range(true_label.shape[1]):
            try:
                roc_auc_temp_mi.append(roc_auc_score(true_label[:, i].cpu().numpy(), pre_label[:, i].cpu().numpy(), average='micro'))
                roc_auc_temp_ma.append(roc_auc_score(true_label[:, i].cpu().numpy(), pre_label[:, i].cpu().numpy(), average='macro'))
                continue
            except:
                continue

        if len(roc_auc_temp_mi)>0:
            roc_auc_mi.append(np.mean(np.array(roc_auc_temp_mi)))
        if len(roc_auc_temp_ma)>0:
            roc_auc_ma.append(np.mean(np.array(roc_auc_temp_ma)))
    pred_batch = torch.cat(pred_list,dim=0)
    label_batch = torch.cat(label_list,dim=0)
    global_roc_mi = []
    global_roc_ma = []
    for i in range(label_batch.shape[1]):
        try:
            global_roc_mi.append(roc_auc_score(label_batch[:, i].cpu().numpy(), pred_batch[:, i].cpu().numpy(), average='micro'))
            global_roc_ma.append(roc_auc_score(label_batch[:, i].cpu().numpy(), pred_batch[:, i].cpu().numpy(), average='macro'))
            continue
        except:
            continue
    
    return np.mean(np.array(roc_auc_mi)), np.mean(np.array(roc_auc_ma)), np.mean(np.array(global_roc_mi)), np.mean(np.array(global_roc_ma)), total_loss / total_examples,\
            emb, ylist, contri, masklist, smileslist 



@torch.no_grad()
def test_reg_mse_attn(loader, gnn_model, args):
    gnn_model.eval()
    mse = []
    emb = []
    contri = []
    masklist = []
    smileslist = []
    ylist = []
    for data in loader:
        data = data.to(args['device'])
        out, embed, contributes, mask, pool_loss = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
        emb.append(embed.cpu())
        contri.append(contributes)
        masklist.append(mask)
        ylist.append(data.y)
        smileslist.append(data.smiles)
    return float(torch.cat(mse, dim=0).mean().sqrt()), emb, ylist, contri, masklist, smileslist

# Train reg MSE
def train_reg_mse(train_loader, gnn_model, optimizer, loss, args):
    if not loss:
        loss = nn.MSELoss()
    gnn_model.train()
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(args['device'])
        optimizer.zero_grad()
        out, embed, contributes, mask, pool_loss = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        nn_loss = loss(data.y, out, embed) + pool_loss
        nn_loss.backward()
        optimizer.step()
        total_loss += float(nn_loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss / total_examples
# Test reg MSE
@torch.no_grad()
def test_reg_mse(loader, gnn_model, args):
    gnn_model.eval()
    mse = []
    for data in loader:
        data = data.to(args['device'])
        out, embed, contributes, mask, pool_loss = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())


# Train anycls CrossEntropy
def train_cls_celoss(train_loader, gnn_model, optimizer, loss, args):
    if not loss:
        loss = nn.CrossEntropyLoss()
    gnn_model.train()
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(args['device'])
        optimizer.zero_grad()
        out, embed, contributes, mask, pool_loss = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        #import ipdb;ipdb.set_trace()
        nn_loss = loss(out, data.y.squeeze().to(torch.long))
        #print(nn_loss)
        nn_loss.backward()
        optimizer.step()
        total_loss += float(nn_loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss / total_examples
# Test 2cls CrossEntropy
@torch.no_grad()
def test_2cls_celoss(loader, gnn_model, args):
    gnn_model.eval()
    roc_auc = []
    for data in loader:
        data = data.to(args['device'])
        out, embed, contributes, mask, pool_loss  = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        score = F.softmax(out, dim=-1)
        pre_label = score[:, 1]
        try:
            roc_auc.append(roc_auc_score(data.y.cpu().numpy(), pre_label.cpu().numpy(), average='macro'))
        except:
            continue
    return np.mean(np.array(roc_auc))

@torch.no_grad()
def test_2cls_celoss_draw_attn(loader, gnn_model, args):
    gnn_model.eval()
    roc_auc = []
    for data in loader:
        data = data.to(args['device'])
        out, embed, contributes, mask, pool_loss  = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0], smilesz=data.smiles)
        score = F.softmax(out, dim=-1)
        pre_label = score[:, 1]
        try:
            roc_auc.append(roc_auc_score(data.y.cpu().numpy(), pre_label.cpu().numpy(), average='macro'))
        except:
            continue
    return np.mean(np.array(roc_auc))

# Train muti 2cls BCELoss
def train_muti2cls_bceloss(train_loader, gnn_model, optimizer, loss, args):
    if not loss:
        loss = nn.BCEWithLogitsLoss()
    gnn_model.train()
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(args['device'])
        optimizer.zero_grad()
        out, embed, contributes, mask, pool_loss  = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])

        nn_loss = loss(out, data.y.to(torch.float))
        nn_loss.backward()
        optimizer.step()
        total_loss += float(nn_loss) * data.num_graphs
        total_examples += data.num_graphs
        #print(sqrt(total_loss / total_examples))
    return total_loss / total_examples

@torch.no_grad()
def test_muti2cls_bceloss(loader, gnn_model, args):
    gnn_model.eval()

    pred_list = []
    label_list = []
    roc_auc_mi = []
    roc_auc_ma = []
    total_loss = total_examples = 0
    for i,data in enumerate(loader):
        roc_auc_temp_mi = []
        roc_auc_temp_ma = []
        data = data.to(args['device'])
        out, embed, contributes, mask, pool_loss = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        loss = nn.BCEWithLogitsLoss()
        nn_loss = loss(out, data.y.squeeze().to(torch.float)).detach().cpu()
        total_loss += float(nn_loss) * data.num_graphs
        total_examples += data.num_graphs

        score = F.sigmoid(out)
        pre_label = score
        true_label = data.y.squeeze().to(torch.long)
        pred_list.append(pre_label)
        label_list.append(true_label)

        # print(true_label,true_label.shape)
        # 1/0
        for i in range(true_label.shape[1]):
            try:
                roc_auc_temp_mi.append(roc_auc_score(true_label[:, i].cpu().numpy(), pre_label[:, i].cpu().numpy(), average='micro'))
                roc_auc_temp_ma.append(roc_auc_score(true_label[:, i].cpu().numpy(), pre_label[:, i].cpu().numpy(), average='macro'))
                continue
            except:
                continue

        if len(roc_auc_temp_mi)>0:
            roc_auc_mi.append(np.mean(np.array(roc_auc_temp_mi)))
        if len(roc_auc_temp_ma)>0:
            roc_auc_ma.append(np.mean(np.array(roc_auc_temp_ma)))
    pred_batch = torch.cat(pred_list,dim=0)
    label_batch = torch.cat(label_list,dim=0)
    global_roc_mi = []
    global_roc_ma = []
    for i in range(label_batch.shape[1]):
        try:
            global_roc_mi.append(roc_auc_score(label_batch[:, i].cpu().numpy(), pred_batch[:, i].cpu().numpy(), average='micro'))
            global_roc_ma.append(roc_auc_score(label_batch[:, i].cpu().numpy(), pred_batch[:, i].cpu().numpy(), average='macro'))
            continue
        except:
            continue
    

    return np.mean(np.array(roc_auc_mi)), np.mean(np.array(roc_auc_ma)), np.mean(np.array(global_roc_mi)), np.mean(np.array(global_roc_ma)), total_loss / total_examples





@torch.no_grad()
def test_cls(loader, gnn_model):
    gnn_model.eval()
    roc_auc = []
    for data in loader:
        data = data.to(fixed_train_args['device'])
        y_hat, embed, contributes, mask = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        roc_auc.append(F.mse_loss(y_hat, data.y, reduction='none').cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())


@torch.no_grad()
def test_muti_task(loader, gnn_model):
    gnn_model.eval()
    roc_auc = []
    for data in loader:
        data = data.to(fixed_train_args['device'])
        y_hat, embed, contributes, mask = gnn_model(data.x.float(), data.edge_index,
                                            data.edge_attr, data.batch, data.fp, data.fp_length[0])
        score = F.softmax(embed, dim=-1)
        pre_label = score[:, 1]
        ture_label = data.y.squeeze().to(torch.long)

        try:
            roc_auc.append(roc_auc_score(data.y.cpu().numpy(), pre_label.cpu().numpy(), average='macro'))
        except:
            continue
    return np.mean(np.array(roc_auc))

@torch.no_grad()
def predict(test_loader, model):
    preds = []
    gnn_model.eval()
    for data in test_loader:
        data = data.to(fixed_train_args['device'])
        atomic_contribs, predictions, embeddings = gnn_model(data.x.float(), data.edge_index,
                                                         data.edge_attr, data.batch, data.fp, data.fp_length[0])
        preds.append(predictions)

    predictions = torch.concat(preds, axis=0)
    return predictions

