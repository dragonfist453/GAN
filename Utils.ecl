IMPORT Python3 as Python;
IMPORT GNN.Tensor;
IMPORT $.Types;
TensData := Tensor.R4.TensData;
LayerSpec := Types.LayerSpec;

EXPORT Utils := MODULE
/** This is an Utility function which eases building the Layer Specifications Dataset to be passed to the train function. 
  * It simply takes the layer definition set and the compile definition string, puts them into the dataset to be used as required.  
  * @param layerDef Set of strings as defined in GNN for layer definition of a sequential model
  * @param compiledef String which has the compilation specifications for a model
  * @return A dataset which has them both together to be accessed and moved easily
  */
	EXPORT DATASET(LayerSpec) makeLayerSpec(SET OF STRING layerDef = [''],
											STRING compiledef = '') := FUNCTION
		layerspecs := DATASET(1, TRANSFORM(LayerSpec,
                            SELF.layerDef := layerDef,
                            SELF.compileDef := compileDef
                            ));
		return layerspecs;
	END;	
END;