
# setup step 3

cd data

# imagenet validation set
mkdir imagenet
cd imagenet
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -q --show-progress
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
echo 'imagenet validation set done...'

# broden
# https://github.com/davidbau/quick-netdissect
#python -m netdissect --download
wget http://netdissect.csail.mit.edu/data/broden1_224.zip
unzip broden1_224.zip
rm broden1_224.zip
rm broden1_224/images/ade20k/*color.png
rm broden1_224/images/ade20k/*object.png
rm broden1_224/images/ade20k/*part_*.png
rm broden1_224/images/dtd/*.png
rm -r broden1_224/images/opensurfaces/  # broden osf is a subset of osf (see below)
rm broden1_224/images/pascal/*.png
echo 'bds done...'

# tinyimagenet
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
echo 'tin done...'

# osf
wget http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz
tar -xvf minc-2500.tar.gz
rm minc-2500.tar.gz
echo 'osf done...'

cd ..
mkdir results
