import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pickle 
import os
import glob
import yaml
import random 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, int(0.5 * input_dim))
        self.bn1 = nn.BatchNorm1d(int(input_dim *0.5))
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(int(0.5 * input_dim), int(input_dim / 6))
        self.bn2 = nn.BatchNorm1d(int(input_dim / 6))
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(int(input_dim / 6), 4096)
        self.fc4 = nn.Linear(4096, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        
        self.fc7_mean = nn.Linear(128, latent_dim)
        self.fc7_log_var = nn.Linear(128, latent_dim)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.elu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.celu(self.fc3(x), alpha=0.95)
        x = F.rrelu(self.fc4(x), lower=-0.2, upper=1.7)
        x = F.leaky_relu(self.fc5(x))
        x= F.elu(self.fc6(x))
        z_mean = self.fc7_mean(x)
        z_log_var = self.fc7_log_var(x)
        return z_mean, z_log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(512, 4096)
        self.fc5 = nn.Linear(4096, int(output_dim / 6))
        self.fc6 = nn.Linear(int(output_dim / 6), int(output_dim / 2))
        self.fc7 = nn.Linear(int(output_dim / 2), output_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        h = F.leaky_relu(self.bn1(self.fc1(z)))
        h = self.dropout1(h)
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.bn3(self.fc3(h)))
        h = self.dropout3(h)
        h = F.celu(self.fc4(h), alpha=0.9)
        h = F.celu(self.fc5(h))
        h = self.fc6(h)
        return torch.tanh(self.fc7(h))

def sampling(z_mean, z_log_var, epsilon=1e-6):
    std = torch.exp(0.5 * z_log_var) + epsilon
    eps = torch.randn_like(std)
    return z_mean + eps * std
    

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = sampling(z_mean, z_log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, z_mean, z_log_var

    def evaluate(self, x):
        self.eval()
        with torch.no_grad():
            z_mean, z_log_var = self.encoder(x)
            z = sampling(z_mean, z_log_var)
            reconstructed_x = self.decoder(z)
        self.train()
        return reconstructed_x, z_mean, z_log_var

def vae_loss(reconstructed_x, x, z_mean, z_log_var, n_maillage, n_input, alpha):
    reconstruction_loss = (alpha / (2 * n_maillage * n_input)) * F.mse_loss(reconstructed_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var + 1e-10))
    return reconstruction_loss + kl_loss , kl_loss , reconstruction_loss


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = sampling(z_mean, z_log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, z_mean, z_log_var

    def evaluate(self, x):
        self.eval()
        with torch.no_grad():
            z_mean, z_log_var = self.encoder(x)
            z = sampling(z_mean, z_log_var)
            reconstructed_x = self.decoder(z)
        self.train()
        return reconstructed_x, z_mean, z_log_var


def train_vae(model, dataloader, n_epochs, n_edge, n_nodes, alpha, OUTPUT_folder,  learning_rate=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model.to(device)
    kl = []
    loss_g =[]
    reconstruction=[]
    
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0
        train_kl = 0
        train_reconstruction =0
        
        for batch_idx, batch in enumerate(dataloader):
            x = batch.to(device).float()
            optimizer.zero_grad()
            reconstructed_x, z_mean, z_log_var = model(x)

            # Verify dimensions
            assert reconstructed_x.shape == x.shape, f"Output shape {reconstructed_x.shape} does not match input shape {x.shape}"
            assert z_mean.shape[1] == model.encoder.fc7_mean.out_features, f"Latent mean shape {z_mean.shape} does not match expected shape"
            assert z_log_var.shape[1] == model.encoder.fc7_log_var.out_features, f"Latent log_var shape {z_log_var.shape} does not match expected shape"
            
            loss, kl_loss, reconstruction_loss = vae_loss(reconstructed_x, x, z_mean, z_log_var, n_edge, n_nodes, alpha)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_kl += kl_loss.item()
            train_reconstruction += reconstruction_loss.item()
            
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item()}')
        
        avg_loss = train_loss / len(dataloader.dataset)
        loss_g.append(avg_loss)
        avg_kl = train_kl / len(dataloader.dataset)
        kl.append(avg_kl)
        avg_recon = train_reconstruction / len(dataloader.dataset)
        reconstruction.append(avg_recon)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss} , Kl_loss = {avg_kl}, reconstruction loss={avg_recon}')
        
        if epoch == 15:
            model_name = OUTPUT_folder+'vae_model_15_modif.pth'
            torch.save(model.state_dict(), model_name)
            print(f'Model saved to {model_name}')
    
        # Créez une figure et des axes pour le tracé
    plt.figure(figsize=(10, 10))

    # Tracez les pertes
    plt.plot(loss_g[1:], label='Total Loss')
    plt.plot(kl[1:], label='KL Loss')
    plt.plot(reconstruction[1:], label='Reconstruction Loss')

    # Ajoutez des labels et un titre
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Epochs')

    # Ajoutez une légende
    plt.legend()
    plt.savefig(OUTPUT_folder+f'Perte_en_fonction_des_epoque_alpha{alphaa}.png')
    print("Perte_modif.png")

    # Save the final model
    final_model_name = OUTPUT_folder+ f"vae_model_final_modif_alpha{alpha}.pth"
    torch.save(model.state_dict(), final_model_name)
    print(f'Model saved to {final_model_name}')


if __name__ == '__main__':
    #
    script_file = '/home/jbetancur/dev/tsi/ssm_fatine/ssm/Generation_deformation_clean/data_filRougeCentraleSupelec/ParamsEntrainementVAE.yaml'
    #
    with open(script_file, 'r') as fid:
        yamlList = yaml.safe_load(fid)
    #
    #getting parameters
    PARAMETERS = yamlList[0]
    INPUT_FOLDER = PARAMETERS['INPUT_FOLDER']
    ext_file = PARAMETERS['extension_file']
    output_folder_VAE= PARAMETERS['output_folder_VAE']
    folder_information_mesh = PARAMETERS['Information_maillage_folder']
    pattern = os.path.join(INPUT_FOLDER, "*"+ ext_file )
    search_pattern = glob.glob(pattern)
    n_file = len(search_pattern)
    print(n_file)
    #
    data = []
    labels =[]
    percentage_edges= []
    distance_hausdorff = []
    different_defomation = ['twist', 'boursouflure', 'retrait','scaling']
    #
    print(INPUT_FOLDER)
    for deformation in different_defomation:
        search_pattern = os.path.join(INPUT_FOLDER, deformation+ "*"+ ext_file )
        matching_files = glob.glob(search_pattern)
        # if len(matching_files) >= n_each:
        #     random_files = random.choices(matching_files, k=n_each)
        # else:
        #     random_files = random.choices(matching_files, k=len(matching_files))

        for file in matching_files : 
            with open(file, 'rb') as file2:
                vector = pickle.load(file2)
                vector = torch.tensor(vector, dtype=torch.float32)  # Convert to float16 tensor
                if torch.isfinite(vector).all():
                    data.append(vector)
                    labels.append(deformation)
                    name_file = os.path.basename(file)
                    with open(folder_information_mesh+name_file, 'rb') as file1:
                        dictonnaire_information = pickle.load(file1)
                        distance_hausdorff_mesh = dictonnaire_information.get('hausdorff_distance')
                        distance_hausdorff.append(distance_hausdorff_mesh)
                        percentage_edges_mesh = dictonnaire_information.get('pourcentage_arrete')
                        percentage_edges.append(percentage_edges_mesh)

                #     name_file = os.path.basename(random_file[i])
                #     if name_file.startswith('twist_'):
                #         labels.append('twist')
                #     elif name_file.startswith('boursouflure_'):
                #         labels.append('blister')
                #     elif name_file.startswith('TC_'):
                #         labels.append('S')  # O pour "Other"
                # else:
                #     print(f"NaN or inf values in file {file_random}, skipping.")
        # except FileNotFoundError:
        #     print(f"FileNotFoundError: File {file_random} not found. Skipping this file.")
        # except Exception as e:
        #     print(f"An error occurred while loading file {file_random}: {e}")
    batch_size = 32
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # Define model, input_dim, latent_dim, and other parameters
    if len(data) > 0:  # Vérifiez si des données sont chargées
        input_dim = data[0].size()[0]  # Obtenir la taille de la première dimension du premier tensor
        print(f"Input dimension: {input_dim}")
    else:
        print("Error: Not enough data loaded. Ensure there are enough valid pickle files.")
    n_epochs = 30
    alphaa = 10**5
    latent_dim = 64

    vae = VAE(input_dim, latent_dim)
    vae.apply(weights_init)

    # Train the VAE model
    train_vae(vae, dataloader, n_epochs, input_dim, n_file, alphaa, output_folder_VAE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize and load the pre-trained VAE model
    vae = VAE(input_dim, latent_dim).to(device)
    path_weight_VAE =  output_folder_VAE+ f"vae_model_final_modif_alpha{alphaa}.pth"
    vae.load_state_dict(torch.load(path_weight_VAE, map_location=device))
    vae.eval()

    # Obtain the latent representations
    with torch.no_grad():
        data_tensor = torch.stack(data).to(device)  # Ensure data is moved to the same device as the model
        z_mean, z_log_var = vae.encoder(data_tensor)
        latents = sampling(z_mean, z_log_var)

    # Convert the latents to numpy array
    latents_numpy = latents.cpu().numpy()

    # Convert the labels to integers for use in the palette
    unique_labels = list(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels])

    # Use t-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents_numpy)

    # Create a color palette for the different classes
    palette = sns.color_palette("hsv", len(unique_labels))

    # print('label', unique_labels )
    # # Visualization
    # plt.figure(figsize=(10, 10))
    # for i, label in enumerate(unique_labels):
    #     idx = np.where(int_labels == i)
    #     plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1], c=np.array(palette[i]).reshape(1, -1), alpha=0.5, label=label)
    # plt.title('Données dans l\'espace latent (réduit à 2D avec t-SNE)')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend()

    # # Save the plot
    # output_filename = 'TSNE_vae_model_final_30.png'
    # plt.savefig(output_filename)
    # print(f"Plot saved to {output_filename}")
    # plt.show()

    # Visualisation des données spécifiques (boursouflure, twit, scale)
    # specific_labels = ['boursouflure', 'twist', 'scaling','retrait']
    plt.figure(figsize=(10, 10))
    for label in different_defomation :
        if label in unique_labels:
            idx = np.where(int_labels == label_to_int[label])
            plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1], c=np.array(palette[label_to_int[label]]).reshape(1, -1), alpha=0.5, label=label)
    plt.title('Données spécifiques dans l\'espace latent (réduit à 2D avec t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(output_folder_VAE+f'TSNE_specific_vae_model1_batch20_modif_alpha{alphaa}.png')
    print("Plot saved to TSNE_specific_vae_model.png")
    plt.show()



    ### Distance d'Hausdorf ###

    plt.figure(figsize=(10, 10))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha =0.5)
    norm = plt.Normalize(vmin=min(distance_hausdorff), vmax=max(distance_hausdorff))
    cmap = plt.cm.viridis

    scatter= plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=distance_hausdorff, cmap=cmap, alpha =0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Distance Hausdorff')
    plt.title('Projection des données d\'entrée dans l\'espace latent (réduit à 2D avec t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(output_folder_VAE+f'TSNE_specific_distancehausdorff_batch20_modif_alpha{alphaa}.png')
    print("Plot saved to TSNE_specific_distancehausdorff.png")
    plt.show()

    ###    Percentage edge  ### 

    plt.figure(figsize=(10, 10))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha =0.5)
    norm = plt.Normalize(vmin=min(percentage_edges), vmax=max(percentage_edges))
    cmap = plt.cm.viridis

    scatter= plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=percentage_edges, cmap=cmap, alpha =0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Distance Hausdorff')
    plt.title('Projection des données d\'entrée dans l\'espace latent (réduit à 2D avec t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(output_folder_VAE+f'TSNE_specific_percentage_edge_batch20_modif_alpha{alphaa}.png')
    print("Plot saved to TSNE_specific_percentage_edge.png")
    plt.show()

    # Visualisation des groupes spécifiques (BT, BC, TC)
    # group_labels = ['BT', 'BC', 'TC']
    # plt.figure(figsize=(10, 10))
    # for label in group_labels:
    #     if label in unique_labels:
    #         idx = np.where(int_labels == label_to_int[label])
    #         plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1], c=np.array(palette[label_to_int[label]]).reshape(1, -1), alpha=0.5, label=label)
    # plt.title('Groupes spécifiques dans l\'espace latent (réduit à 2D avec t-SNE)')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend()
    # plt.savefig('TSNE_groups_vae_model_model1.png')
    # print("Plot saved to TSNE_groups_vae_model.png")
    # plt.show()

    # # Visualisation de toutes les données
    # plt.figure(figsize=(10, 10))
    # for i, label in enumerate(unique_labels):
    #     idx = np.where(int_labels == i)
    #     plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1], c=np.array(palette[i]).reshape(1, -1), alpha=0.5, label=label)
    # plt.title('Toutes les données dans l\'espace latent (réduit à 2D avec t-SNE)')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend()
    # output_filename = 'TSNE_all_vae_model1.png'
    # plt.savefig(output_filename)
    # print(f"Plot saved to {output_filename}")
    # plt.show()

    # Use PCA to reduce dimension to 2 
    acp = PCA(n_components=2)
    acp_latent = acp.fit_transform(latents_numpy)

    plt.figure(figsize=(10, 10))
    for label in different_defomation :
        if label in unique_labels:
            idx = np.where(int_labels == label_to_int[label])
            plt.scatter(acp_latent[idx, 0], acp_latent[idx, 1], c=np.array(palette[label_to_int[label]]).reshape(1, -1), alpha=0.5, label=label)
    plt.title('Données spécifiques dans l\'espace latent (réduit à 2D avec t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(output_folder_VAE+f'ACP_specific_vae_model_batch20_modif_alpha{alphaa}.png')
    print("Plot saved to ACP_specific_vae_model.png")
    plt.show()

    ### Distance d'Hausdorf ###

    plt.figure(figsize=(10, 10))
    plt.scatter(acp_latent[:, 0], acp_latent[:, 1], alpha =0.5)
    norm = plt.Normalize(vmin=min(distance_hausdorff), vmax=max(distance_hausdorff))
    cmap = plt.cm.viridis

    scatter= plt.scatter(acp_latent[:, 0], acp_latent[:, 1], c=distance_hausdorff, cmap=cmap, alpha =0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Distance Hausdorff')
    plt.title('Projection des données d\'entrée dans l\'espace latent (réduit à 2D avec t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(output_folder_VAE+f'ACP_specific_distancehausdorff_batch20_modif_alpha{alphaa}.png')
    print("Plot saved to TSNE_specific_distancehausdorff.png")
    plt.show()

    ###    Percentage edge  ### 

    plt.figure(figsize=(10, 10))
    plt.scatter(acp_latent[:, 0], acp_latent[:, 1], alpha =0.5)
    norm = plt.Normalize(vmin=min(percentage_edges), vmax=max(percentage_edges))
    cmap = plt.cm.viridis

    scatter= plt.scatter(acp_latent[:, 0], acp_latent[:, 1], c=percentage_edges, cmap=cmap, alpha =0.5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Percentage edge')
    plt.title('Projection des données d\'entrée dans l\'espace latent (réduit à 2D avec t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(output_folder_VAE+f'ACP_specific_percentage_edge_batch20_modif_alpha{alphaa}.png')
    print("Plot saved to TSNE_specific_percentage_edge.png")
    plt.show()

    # # Visualisation des groupes spécifiques (BT, BC, TC)
    # plt.figure(figsize=(10, 10))
    # for label in group_labels:
    #     if label in unique_labels:
    #         idx = np.where(int_labels == label_to_int[label])
    #         plt.scatter(acp_latent[idx, 0], acp_latent[idx, 1], c=np.array(palette[label_to_int[label]]).reshape(1, -1), alpha=0.5, label=label)
    # plt.title('Groupes spécifiques dans l\'espace latent (réduit à 2D avec t-SNE)')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend()
    # plt.savefig('ACP_groups_vae_model_model.png')
    # print("Plot saved to TSNE_groups_vae_model.png")
    # plt.show()

    # # Visualisation de toutes les données
    # plt.figure(figsize=(10, 10))
    # for i, label in enumerate(unique_labels):
    #     idx = np.where(int_labels == i)
    #     plt.scatter(acp_latent[idx, 0], acp_latent[idx, 1], c=np.array(palette[i]).reshape(1, -1), alpha=0.5, label=label)
    # plt.title('Toutes les données dans l\'espace latent (réduit à 2D avec t-SNE)')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.legend()
    # output_filename = 'ACP_all_vae_model.png'
    # plt.savefig(output_filename)
    # print(f"Plot saved to {output_filename}")
    # plt.show()
