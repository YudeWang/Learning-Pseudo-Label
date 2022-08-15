import time
import torch
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from torch.utils.data import DataLoader

class BatchThreader:
	def __init__(self, func, args_list, processes):
		self.n_tasks = len(args_list)
		self.processes = processes
		self.pool = ThreadPool(processes=processes)
		self.async_result = []

		self.func = func
		self.left_args_list = args_list

		# initial work
		self.__start_works(min(processes, self.n_tasks))


	def __start_works(self, times):
		for _ in range(times):
			args = self.left_args_list.pop(0)
			self.async_result.append(
				self.pool.apply_async(self.func, args))

	
	def pop_results(self):
		rtn = []
		time_list = []
		for _ in range(self.n_tasks):
			item = self.async_result.pop(0).get()
			rtn.append(item[0])
			time_list.append(item[1])
			if len(self.left_args_list) > 0:
				args = self.left_args_list.pop(0)	
				self.async_result.append(self.pool.apply_async(self.func, args))

#		rtn = [self.async_result.pop(0).get()
#			for _ in range(n_fetch)]

		return rtn, time_list

def multi_gpu_test(model, dataloader, prepare_func, inference_func, collect_func, save_step_func=None):
	model.eval()
	n_gpus = torch.cuda.device_count()
	net_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
	collect_list = []
	with torch.no_grad():
		for i_batch, sample in enumerate(dataloader):
			name = sample['name']
			image_msf = prepare_func(sample)

			def _work(i, img):
				with torch.no_grad():
					with torch.cuda.device(i%n_gpus):
						start_time = time.time()
						result = inference_func(net_replicas[i%n_gpus], img.cuda())
						end_time = time.time()
						t = end_time - start_time
						return (result,t)

			thread_pool = BatchThreader(_work, list(enumerate(image_msf)), processes=4)
			result_list, time_list = thread_pool.pop_results()
			result_item = collect_func(result_list, sample) 
			result_sample = {'predict': result_item, 'name':name[0]}
			print('%d/%d'%(i_batch,len(dataloader)))

			if save_step_func is not None:
				save_step_func(result_sample)
			else:
				collect_list.append(result_sample)
	return collect_list

def single_gpu_test(model, dataloader, prepare_func, inference_func, collect_func, save_step_func=None):
	model.eval()
	n_gpus = torch.cuda.device_count()
	#assert n_gpus == 1
	collect_list = []
	total_num = len(dataloader)
	with tqdm(total=total_num) as pbar:
		with torch.no_grad():
			for i_batch, sample in enumerate(dataloader):
				name = sample['name']
				image_msf = prepare_func(sample)
				result_list = []
				for img in image_msf:
					result = inference_func(model, img.cuda())	
					result_list.append(result)
				result_item = collect_func(result_list, sample)
				result_sample = {'predict': result_item, 'name':name[0]}
				#print('%d/%d'%(i_batch,len(dataloader)))
				pbar.set_description('Processing')
				pbar.update(1)
				time.sleep(0.001)

				if save_step_func is not None:
					save_step_func(result_sample)
				else:
					collect_list.append(result_sample)
	return collect_list

def single_gpu_plot(model, dataloader, prepare_func, inference_func, collect_func, save_step_func=None):
	model.eval()
	n_gpus = torch.cuda.device_count()
	#assert n_gpus == 1
	collect_list = []
	total_num = len(dataloader)
	with tqdm(total=total_num) as pbar:
		with torch.no_grad():
			for i_batch, sample in enumerate(dataloader):
				name = sample['name']
				image_msf = prepare_func(sample)
				result_list = []
				for img in image_msf:
					result = inference_func(model, img.cuda())	
					result_list.append(result)
				result_item = collect_func(result_list, sample)
				result_sample = {'predict': result_item, 'name':name[0]}
				#print('%d/%d'%(i_batch,len(dataloader)))
				pbar.set_description("Processing")
				pbar.update(1)
				time.sleep(0.001)				

				if save_step_func is not None:
					save_step_func(result_sample)
				else:
					collect_list.append(result_sample)
	return collect_list
