import os

no_images = 600
noise_epsilons = [0.5, 1.0]

os.system(f'ipython AdversarialImageNet.py -- --command measure_accuracy --no_images 50000')
#os.system(f'ipython AdversarialImageNet.py -- --command generate_conventional_from_validation --no_images {no_images}')
#os.system(f'ipython AdversarialImageNet.py -- --command generate_conventional_from_gan --no_images {no_images}')

if False:
    for noise_epsilon in noise_epsilons:
        os.system(f'ipython AdversarialImageNet.py -- --command generate_bounded --no_images {no_images} --noise_epsilon {noise_epsilon} --bounded_search_rho 0.1')
        os.system(f'ipython AdversarialImageNet.py -- --command generate_minimum --no_images {no_images} --noise_epsilon {noise_epsilon}')
