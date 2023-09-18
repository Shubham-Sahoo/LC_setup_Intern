import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import InMemoryDataset 
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, GCNConv, GatedGraphConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn.metrics import mean_squared_error as mse
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from devices.set.models.nmodel import Node_feat, Edge_feat, Net, Gat_Net, mod_call

device = torch.device('cpu')
#model = Gat_Net().to(device)
model = mod_call().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.MSELoss()

def lc_train(im_t1,im_t2,policy_y,edge_v,edge_t1,edge_t2):
    
    data_t_1 = im_t1.T
    data_t_2 = im_t2.T
    C,F = data_t_1.shape
    
    #print(edge_t1.shape)
    #edge_t1 = np.amax(edge_t1,axis=1)
    #edge_t2 = np.amax(edge_t2,axis=1)
    #print(edge_t1.shape)
    edge_t1 = np.reshape(edge_t1,(-1,4))
    edge_t2 = np.reshape(edge_t2,(-1,4))
    #print(edge_t1.shape)
    data_tr = np.vstack((data_t_1,data_t_2))
    data = np.mean(data_tr,axis=1)
    d_policy = np.hstack((policy_y,policy_y))
    #print(data.shape)
    #print(d_policy.shape)
    data = np.resize(data,(data.shape[0],1))
    d_policy = np.resize(d_policy,(d_policy.shape[0],1))
    data = np.hstack((data,d_policy))
    #print(data.shape)
    node_features = torch.FloatTensor(data)

    #print(node_features.shape)
    
    # Horizontal Edges
    target_nodes_h1 = [i+1 for i in range(int(len(node_features)/2)-1)]                 # For t-1 Nodes
    source_nodes_h1 = [i for i in range(int(len(node_features)/2)-1)]

    target_nodes_h2 = [i for i in range(int(len(node_features)/2)-1)]                   # For t-1 Nodes
    source_nodes_h2 = [i+1 for i in range(int(len(node_features)/2)-1)]
 
    target_nodes_h3 = [i+C+1 for i in range(int(len(node_features)/2)-1)]               # For t-2 Nodes
    source_nodes_h3 = [i+C for i in range(int(len(node_features)/2)-1)]

    target_nodes_h4 = [i+C for i in range(int(len(node_features)/2)-1)]                 # For t-2 Nodes
    source_nodes_h4 = [i+C+1 for i in range(int(len(node_features)/2)-1)]

    #print(target_nodes_h)
    
    # Vertical Edges
    target_nodes_v1 = [j+C for j in range(int(len(node_features)/2))]
    source_nodes_v1 = [j for j in range(int(len(node_features)/2))]
   
    target_nodes_v2 = [j for j in range(int(len(node_features)/2))]
    source_nodes_v2 = [j+C for j in range(int(len(node_features)/2))]

    source_nodes = np.hstack((source_nodes_h1,source_nodes_h2,source_nodes_h3,source_nodes_h4,source_nodes_v1,source_nodes_v2))
    target_nodes = np.hstack((target_nodes_h1,target_nodes_h2,target_nodes_h3,target_nodes_h4,target_nodes_v1,target_nodes_v2))
    

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    #print(len(edge_index[0])-len(edge_attr))
    #print(edge_index.shape)
    edge_attr = np.vstack((edge_t1,edge_t1,edge_t2,edge_t2,edge_v,edge_v))  
    #print(edge_attr.shape)
    edge_attr = torch.FloatTensor(edge_attr)
    #print(edge_index)


    x = node_features
    y = torch.FloatTensor(np.hstack((policy_y,policy_y)))
    #print(y.shape)
    #print(y2)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    train_loader = DataLoader([data],batch_size=1)
    #for data in train_loader:
    #    print(data.y)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    #checkpoint = torch.load("/home/shubham/LC-SET/devices/set/weight")
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #loss = checkpoint['loss']

    def train(train_loader,len_tot,C):
        model.train()

        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            #optimizer.zero_grad()
            #print(list(model.parameters())[0].grad)
            output_x,output_edge = model(data)
            #print(output_x.requires_grad)
            #print(output_x.shape)
            #print(output_edge.shape)
            #out = torch.split(output,C)
            #print(len(output))
            label = data.y.to(device)
            #print(len(label))
            loss = crit(output_x[:,1], label)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
        return loss_all / len_tot

    def evaluate(train_loader,C):
        model.eval()

        predictions = []
        labels = []

        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)
                mpred_x,mpred_edge = model(data)
                pred_x = mpred_x.detach().cpu().numpy()
                pred_edge = mpred_edge.detach().cpu().numpy()
                label = data.y.detach().cpu().numpy()
                predictions.append(pred_x[:,1])
                labels.append(label)

        predictions = np.hstack(predictions)
    
        #print("labels")
        labels = np.hstack(labels)
        #print(labels.shape)
        predictions = np.reshape(predictions,labels.shape)
        #predictions = [predictions>0.5]
        #print(predictions)
        #return mse(labels, predictions)
        print(mse(labels, predictions))
        return np.array(predictions)

    for epoch in range(2):
        
        loss = train(train_loader,len(data_tr),C)
        train_pred = evaluate(train_loader,C)
        #print(model.parameters())
        #w = model.conv1.weight.data.numpy()
        #print(w)
        #val_acc = evaluate(val_loader)    
        #test_acc = evaluate(test_loader)
        #print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}'.
        #  format(epoch, loss, train_pred))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, "/home/shubham/LC-SET/devices/set/weight")
    return np.array(train_pred[C:])

data_tr = np.random.rand(480,640) #C,F
pol = np.random.rand(640)
edge = np.random.rand(640,4)
edge_t = np.random.rand(639,4)
f = lc_train(data_tr,data_tr,pol,edge,edge_t,edge_t)




