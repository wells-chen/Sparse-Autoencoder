function [patches] = loadData(patchsize_value)
%  load([datasetpath,'/', 'smaller_dataset.mat'])
%   traindata = loadMNISTImages([datasetpath,'/','train-images-idx3-ubyte']);
   addpath MNIST/
   traindata = loadMNISTImages('train-images.idx3-ubyte');
   patchsize = patchsize_value;  % we'll use 8x8 patches 
   numpatches = 100000;
   patches = zeros(patchsize*patchsize, numpatches);
   IMAGES=zeros(28,28,size(traindata,2));
   for i=1:size(traindata,2)
       IMAGES(:,:,i)=reshape(traindata(:,i),28,28);
   end
   count=0;
for i=3:size(traindata,2)
   if (count>=numpatches)
     break;
   end
   for j=1:2:28/patchsize
     if (count>=numpatches)
      break;
     end
     for k=1:2:28/patchsize
	      if (count>=numpatches)
             break;
          end
	    patches(:,count+1)=reshape(IMAGES((j-1)*patchsize+1:(j)*patchsize,(k-1)*patchsize+1:(k)*patchsize,i),patchsize*patchsize,1);
        if (sum(patches(:,count+1))~=0)
            count=count+1;
        end
	 end
   end
end
end


% Helper function taken from http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
function images = loadMNISTImages(filename)
                                %loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
                                %the raw MNIST images
  fp = fopen(filename, 'rb');
  assert(fp ~= -1, ['Could not open ', filename, '']);
  magic = fread(fp, 1, 'int32', 0, 'ieee-be');
  assert(magic == 2051, ['Bad magic number in ', filename, '']);
  numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
  numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
  numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
  images = fread(fp, inf, 'unsigned char');
  images = reshape(images, numCols, numRows, numImages);
  images = permute(images,[2 1 3]);
  fclose(fp);
                                % Reshape to #pixels x #examples
  images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
                                % Convert to double and rescale to [0,1]
  images = double(images) / 255;
end

% Helper function taken from http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
function labels = loadMNISTLabels(filename)
                                %loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
                                %the labels for the MNIST images
  fp = fopen(filename, 'rb');
  assert(fp ~= -1, ['Could not open ', filename, '']);
  magic = fread(fp, 1, 'int32', 0, 'ieee-be');
  assert(magic == 2049, ['Bad magic number in ', filename, '']);
  numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
  labels = fread(fp, inf, 'unsigned char');
  assert(size(labels,1) == numLabels, 'Mismatch in label count');
  fclose(fp);
end

