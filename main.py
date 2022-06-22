from data_handler import DATA
from trainer import Trainer
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--niters', type=int, default=50000, help='Number of training iterations')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--z_size', type=int, default=64, help='Latent space dimension')
	parser.add_argument('--filter', type=int, default=128, help='Input CNN filter')
	parser.add_argument('--n_res_block', type=int, default=4, help='How Many Res Blocks')
	parser.add_argument('--n_res_channel', type=int, default=8, help='ResBlock internal size')
	parser.add_argument('--K', type=int, default=256, help='Codebook Size')
	parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate G')
	parser.add_argument('--data_path', type=str, default='../3D-GAN/train_lidc_128.npz',help='Path to data.')
	parser.add_argument('--val_path', type=str, default='../3D-GAN/test_lidc_128.npz',help='Path to validation data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--steps_per_log', type=int, default=10, help='Output Iterations')
	parser.add_argument('--steps_per_img_log', type=int, default=50, help='Image Save Iterations')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--load_params', type=bool, default=False, help='Load Parameters form pickle in log dir')
	params = parser.parse_args()
	print(params)
	
	dataset_train = DATA(path=params.data_path)
	dataset_val = DATA(path=params.val_path)

	trainer = Trainer(dataset_train, dataset_val, params=params)
	trainer.train()

if __name__ == '__main__':
	main()
