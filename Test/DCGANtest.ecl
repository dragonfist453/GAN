IMPORT Python3 AS Python;
IMPORT GNN.Tensor;
IMPORT Std.System.Log AS Syslog;
IMPORT Std;
IMPORT $.^ as GAN;
IMPORT GAN.GAN;
IMPORT $.^.Utils;
IMPORT GNN.GNNI;
IMPORT $.^.Types;
IMPORT GNN.Image;
t_Tensor := Tensor.R4.t_Tensor;
TensData := Tensor.R4.TensData;

#option('outputLimit',2000);
RAND_MAX := POWER(2,8) - 1;
RAND_MAX_2 := RAND_MAX / 2;

//Train data definitions
latentDim := 100;
batchSize := 100;
numEpochs := 2000;
outputRows := 5;
outputCols := 5;
numRecords := 60000;

//Take MNIST dataset using GNN.Image module
mnist_train_images := Image.MNIST.Get_train_images('~test::mnist_train_images')[..numRecords];

//Extract dimensions
imgDims := mnist_train_images[1].imgDims;
imgRows := imgDims[1];
imgCols := imgDims[2];
imgChannels := imgDims[3];

//Convert image to tensor with GNN.Image module
trainX0 := Image.ImgtoTens(mnist_train_images); 
                                           
//Builds tensors for the neural network
trainX := Tensor.R4.MakeTensor([0, imgRows, imgCols, imgChannels], trainX0); 

//GENERATOR
//Generator model definition information
ldef_generator := ['''layers.Input(shape=(100,))''',
                        '''layers.Dense(128 * 7 * 7, activation="relu", input_dim=100)''',
                        '''layers.Reshape((7, 7, 128))''',    
                        '''layers.UpSampling2D()''',
                        '''layers.Conv2D(128, kernel_size=3, padding="same")''',
                        '''layers.BatchNormalization(momentum=0.8)''',
                        '''layers.Activation("relu")''',
                        '''layers.UpSampling2D()''',
                        '''layers.Conv2D(64, kernel_size=3, padding="same")''',
                        '''layers.BatchNormalization(momentum=0.8)''',
                        '''layers.Activation("relu")''',
                        '''layers.Conv2D(1, kernel_size=3, padding="same")''',
                        '''layers.Activation("tanh")''',
                        '''layers.Reshape((28,28,1))'''];
            
//DISCRIMINATOR
//Discriminator model definition information
ldef_discriminator := ['''layers.Input(shape=(28, 28, 1))''',
                            '''layers.Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 1), padding="same")''',
                            '''layers.LeakyReLU(alpha=0.2)''',
                            '''layers.Dropout(0.25)''',
                            '''layers.Conv2D(64, kernel_size=3, strides=2, padding="same")''',
                            '''layers.ZeroPadding2D(padding=((0,1),(0,1)))''',
                            '''layers.BatchNormalization(momentum=0.8)''',
                            '''layers.LeakyReLU(alpha=0.2)''',
                            '''layers.Dropout(0.25)''',
                            '''layers.Conv2D(128, kernel_size=3, strides=2, padding="same")''',
                            '''layers.BatchNormalization(momentum=0.8)''',
                            '''layers.LeakyReLU(alpha=0.2)''',
                            '''layers.Dropout(0.25)''',
                            '''layers.Conv2D(256, kernel_size=3, strides=1, padding="same")''',
                            '''layers.BatchNormalization(momentum=0.8)''',
                            '''layers.LeakyReLU(alpha=0.2)''',
                            '''layers.Dropout(0.25)''',
                            '''layers.Flatten()''',
                            '''layers.Dense(1, activation="sigmoid")'''];

//Compile string for both generator and discriminator
compiledef := '''compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';


//Get generator and discriminator models after training
myDCGAN := GAN.Train(trainX, ldef_generator, ldef_discriminator, compiledef, batchSize, numEpochs, latentDim);

generator := myDCGAN.Generator;
discriminator := myDCGAN.Discriminator;

//Random set of normal data
random_data := DATASET(outputRows*OutputCols*batchSize, TRANSFORM(TensData,
                        SELF.indexes := [(COUNTER-1) DIV latentDim + 1, (COUNTER-1)%latentDim + 1],
                        SELF.value := ((RANDOM() % RAND_MAX) / RAND_MAX_2) - 1));

//Noise tensor
train_noise := Tensor.R4.MakeTensor([0,latentDim], random_data);

//Predict an image from noise
generated := GNNI.Predict(generator, train_noise);

//To get generator data as output
gen_data := Tensor.R4.GetData(generated);

//Outputs the tensor onto a file, so the graphs don't repeat
OUTPUT(gen_data, ,'~GAN::output_tensdata', OVERWRITE);

//Convert from tensor data to images by taking from file
outputImage := Image.TenstoImg(DATASET('~GAN::output_tensdata',TensData,FLAT ));

//Convert image data to jpg format to despray
mnistgrid := Image.OutputGrid(outputImage, outputRows, outputCols, 'Epoch_'+(String)numEpochs);

//Output the grid image to despray as a PNG using prefix filename,filesize
img_out := OUTPUT(mnistgrid, ,'~GAN::output_image', OVERWRITE);

//Despray variables
serv := 'server=http://172.16.2.240:8010 ';
over := 'overwrite=1 ';
action  := 'action=despray ';
dstip   := 'dstip=172.16.2.240 ';
dstfile := 'dstfile=/var/lib/HPCCSystems/mydropzone/*.png ';
srcname := 'srcname=~gan::output_image ';
splitprefix := 'splitprefix=filename,filesize ';
cmd := serv + over + action + dstip + dstfile + srcname + splitprefix;

//Despraying image onto landing zone
despray_image := STD.File.DfuPlusExec(cmd);
SEQUENTIAL(img_out, despray_image);

//Get the weights of the trained generator
weights := GNNI.GetWeights(generator);

//Store the info of the model for predictions
modInfo := DATASET(1, TRANSFORM(Types.ModelInfo,
                            SELF.layerspec := Utils.makeLayerSpec(ldef_generator, compiledef),
                            SELF.modWeights := weights,
                            SELF.desprayCommand := cmd,
                            SELF.outputRows := outputRows,
                            SELF.outputCols := outputCols,
                            SELF.batchSize := batchSize,
                            SELF.latentDim := latentDim,
                            SELF.numEpochs := numEpochs
                            ));

OUTPUT(modInfo, ,'~DCGAN::GeneratorInfo', OVERWRITE);