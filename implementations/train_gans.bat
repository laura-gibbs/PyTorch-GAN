@REM python dcgan/dcgan.py --n_epochs 200 --tile_size 16 --overlap 0
@REM python wgan_gp/wgan_gp.py --n_epochs 80 --tile_size 16 --overlap 0
@REM python dcgan/dcgan.py --n_epochs 80 --tile_size 24
@REM python wgan_gp/wgan_gp.py --n_epochs 80 --tile_size 24
@REM python dcgan/dcgan.py --n_epochs 200 --tile_size 32
@REM python wgan_gp/wgan_gp.py --n_epochs 80 --tile_size 32
@REM python dcgan/dcgan.py --n_epochs 200 --tile_size 40
@REM python wgan_gp/wgan_gp.py --n_epochs 80 --tile_size 40
@REM python dcgan/dcgan.py --n_epochs 200 --tile_size 64 --overlap 32
@REM python wgan_gp/wgan_gp.py --n_epochs 80 --tile_size 64 --overlap 32

@REM python dcgan/dcgan.py --n_epochs 350 --tile_size 16 --overlap 0 --cs
@REM python wgan_gp/wgan_gp.py --n_epochs 350 --tile_size 16 --overlap 0 --cs
python dcgan/dcgan.py --n_epochs 350 --tile_size 24 --cs
python wgan_gp/wgan_gp.py --n_epochs 350 --tile_size 24 --cs
python dcgan/dcgan.py --n_epochs 350 --tile_size 32 --cs
python wgan_gp/wgan_gp.py --n_epochs 350 --tile_size 32 --cs
python dcgan/dcgan.py --n_epochs 350 --tile_size 40 --cs
python wgan_gp/wgan_gp.py --n_epochs 350 --tile_size 40 --cs
@REM python dcgan/dcgan.py --n_epochs 350 --tile_size 64 --overlap 32 --cs
@REM python wgan_gp/wgan_gp.py --n_epochs 350 --tile_size 64 --overlap 32 --cs

