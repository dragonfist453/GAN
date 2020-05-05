IMPORT GNN.Tensor;
t_Tensor := Tensor.R4.t_Tensor;

EXPORT Types := MODULE
EXPORT LayerSpec := RECORD
    SET of STRING layerDef;
    STRING compileDef;
END;

/**
  * This Record type is very useful for saving and checking the generator output again and again in the Test/predictTest.ecl file.  
  * It must be made sure though that the steps followed in saveTest.ecl be implemented at the end of every file where GAN is trained. 
  */
EXPORT ModelInfo := RECORD
    DATASET(LayerSpec) layerspec;
    DATASET(t_Tensor) modWeights;
    STRING desprayCommand;
    UNSIGNED outputRows;
    UNSIGNED outputCols;
    UNSIGNED batchSize;
    UNSIGNED latentDim;
    UNSIGNED numEpochs;
END;
END;