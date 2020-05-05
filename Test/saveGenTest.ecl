IMPORT GNN.GNNI;
IMPORT GAN.Types;
IMPORT GAN.Utils;

//Despray variables
serv := 'server=localhost:8010 ';
over := 'overwrite=1 ';
action  := 'action=despray ';
dstip   := 'dstip=192.168.86.149 ';
dstfile := 'dstfile=/var/lib/HPCCSystems/mydropzone/*.png ';
srcname := 'srcname=~gan::output_image ';
splitprefix := 'splitprefix=filename,filesize ';
cmd := serv + over + action + dstip + dstfile + srcname + splitprefix;

//Required variables
outputRows := 5;
outputCols := 5;
batchSize := 100;
latentDim := 100;
numEpochs := 1;

//GENERATOR
//Generator model definition information
ldef_generator := ['''layers.Input(shape=(100,))''',
                '''layers.Dense(256, input_dim=100)''',
                '''layers.LeakyReLU(alpha=0.2)''',    
                '''layers.BatchNormalization(momentum=0.8)''',
                '''layers.Dense(512)''',
                '''layers.LeakyReLU(alpha=0.2)''',
                '''layers.BatchNormalization(momentum=0.8)''',
                '''layers.Dense(1024)''',
                '''layers.LeakyReLU(alpha=0.2)''',
                '''layers.BatchNormalization(momentum=0.8)''',
                '''layers.Dense(784,activation='tanh')''',
                '''layers.Reshape((28,28,1))'''];
            
compiledef_generator := '''compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))''';

session := GNNI.GetSession();

generator := GNNI.DefineModel(session, ldef_generator, compiledef_generator);

weights := GNNI.GetWeights(generator);

modInfo := DATASET(1, TRANSFORM(Types.ModelInfo,
                            SELF.layerspec := Utils.makeLayerSpec(ldef_generator, compiledef_generator),
                            SELF.modWeights := weights,
                            SELF.desprayCommand := cmd,
                            SELF.outputRows := outputRows,
                            SELF.outputCols := outputCols,
                            SELF.batchSize := batchSize,
                            SELF.latentDim := latentDim,
                            SELF.numEpochs := numEpochs
                            ));

OUTPUT(modInfo, ,'~GAN::GeneratorInfo', OVERWRITE);                            