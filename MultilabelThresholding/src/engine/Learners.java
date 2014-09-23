package engine;

import java.util.Arrays;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.SelectedTag;
import weka.core.SerializationHelper;
import weka.core.Tag;
import weka.core.neighboursearch.LinearNNSearch;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.neural.MMPLearner;
import mulan.classifier.transformation.AdaBoostMH;
import mulan.classifier.transformation.EnsembleOfClassifierChains;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
//import mulan.evaluation.Evaluator;
//import mulan.evaluation.MultipleEvaluation;

public class Learners {
	
	/** The filter to apply to the training data: Normalzie */
	public static final int FILTER_NORMALIZE = 0;
	/** The filter to apply to the training data: Standardize */
	public static final int FILTER_STANDARDIZE = 1;
	/** The filter to apply to the training data: None */
	public static final int FILTER_NONE = 2;
	/** The filter to apply to the training data */
	public static final Tag[] TAGS_FILTER =
	{
		new Tag(FILTER_NORMALIZE, "Normalize training data"),
		new Tag(FILTER_STANDARDIZE, "Standardize training data"),
		new Tag(FILTER_NONE, "No normalization/standardization"),
	};

	private MultiLabelInstances dataset;
	
	private RAkEL learner1_1;
	private double result1_1;
	private RAkEL learner1_2;
	private double result1_2;
	private RAkEL learner1_3;
	private double result1_3;
	
	private EnsembleOfClassifierChains learner2_1;
	private double result2_1;
	private EnsembleOfClassifierChains learner2_2;
	private double result2_2;
	private EnsembleOfClassifierChains learner2_3;
	private double result2_3;
	
	private MMPLearner learner3;
	private double result3;
	
	// Ranking by pairwise comparison
	private MMPLearner learner4;
	private double result4 = -1;
	
	private AdaBoostMH learner5;
	private double result5;
	
//	private int folds = 2;
	private MultiLabelInstances test;
	
	private Boolean saveModels;
	
//	public Learners( MultiLabelInstances dataset, int folds ) throws Exception {
	public Learners( MultiLabelInstances dataset, MultiLabelInstances test, Boolean saveModels ) throws Exception {
		this.saveModels = saveModels;
		
		this.dataset = dataset;
		
//		this.folds = folds;
		this.test = test;
		
		// Base classifiers
		IBk knn = new IBk();
		knn.setKNN(1);
		knn.setWindowSize(0);
		////
		LinearNNSearch lnns = new LinearNNSearch();
		//
		EuclideanDistance df = new EuclideanDistance();
		df.setAttributeIndices("first-last");
		//
		lnns.setDistanceFunction(df);
		////
		knn.setNearestNeighbourSearchAlgorithm(lnns);
		this.learner1_1 = new RAkEL(new LabelPowerset(knn));
		
		SMO svm = new SMO();
		svm.setC(1.0);
		svm.setToleranceParameter(0.001);
		svm.setEpsilon( 1.0 * Math.pow(10,-12) );
		svm.setFilterType( new SelectedTag(FILTER_NORMALIZE, TAGS_FILTER) );
		svm.setNumFolds(-1);
		svm.setRandomSeed(1);
		////
		PolyKernel ker = new PolyKernel();
		ker.setCacheSize(250007);
		ker.setExponent(1.0);
		svm.setKernel(ker);
		////
		this.learner1_2 = new RAkEL(new LabelPowerset(svm));
		
		J48 tree = new J48();
		tree.setConfidenceFactor((float) 0.25);
		tree.setMinNumObj(2);
		this.learner1_3 = new RAkEL(new LabelPowerset(tree));
		
		this.learner2_1 = new EnsembleOfClassifierChains( knn, 10, true, true ); // new MLkNN();
		this.learner2_2 = new EnsembleOfClassifierChains( svm, 10, true, true );
		this.learner2_3 = new EnsembleOfClassifierChains( tree, 10, true, true );
		
		// Ranking
		this.learner3 = new MMPLearner();
		
		//ranking by pairwise comparison!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		this.learner4 = new MMPLearner();
		
		//AdaBoost
		this.learner5 = new AdaBoostMH();
	}
	
	public void learn( String name ) throws Exception {
		String[] names;
		String a;
		String b;
		System.out.println("Building the models...");
		
		System.out.println("1...");
        learner1_1.build(dataset);
        if(saveModels){
        	names = learner1_1.getClass().toString().split("\\.");
        	a = names[names.length-1];
        	names = learner1_1.getBaseLearner().getClass().toString().split("\\.");
        	b = names[names.length-1];
        	SerializationHelper.write(a + "-" + b + "-" + name + ".model", learner1_1);
        }
		
        /*
        System.out.println("2...");
        learner1_2.build(dataset);
        if(saveModels){
        	names = learner1_2.getClass().toString().split("\\.");
        	a = names[names.length-1];
        	names = learner1_2.getBaseLearner().getClass().toString().split("\\.");
        	b = names[names.length-1];
        	SerializationHelper.write(a + "-" + b + "-" + name + ".model", learner1_2);
        }
        
        
        System.out.println("3...");
        learner1_3.build(dataset);
        if(saveModels){
        	names = learner1_3.getClass().toString().split("\\.");
        	a = names[names.length-1];
        	names = learner1_3.getBaseLearner().getClass().toString().split("\\.");
        	b = names[names.length-1];
        	SerializationHelper.write(a + "-" + b + "-" + name + ".model", learner1_3);
        }
        
        
        System.out.println("4...");
        learner2_1.build(dataset);
        if(saveModels){
        	names = learner2_1.getClass().toString().split("\\.");
        	a = names[names.length-1];
        	names = learner2_1.getBaseClassifier().getClass().toString().split("\\.");
        	b = names[names.length-1];
        	SerializationHelper.write(a + "-" + b + "-" + name + ".model", learner2_1);
        }
        	
        
        System.out.println("5...");
        learner2_2.build(dataset);
        if(saveModels){
        	names = learner2_2.getClass().toString().split("\\.");
        	a = names[names.length-1];
        	names = learner2_2.getBaseClassifier().getClass().toString().split("\\.");
        	b = names[names.length-1];
        	SerializationHelper.write(a + "-" + b + "-" + name + ".model", learner2_2);
        }
        
        
        System.out.println("6...");
        learner2_3.build(dataset);
        if(saveModels){
        	names = learner2_3.getClass().toString().split("\\.");
        	a = names[names.length-1];
        	names = learner2_3.getBaseClassifier().getClass().toString().split("\\.");
        	b = names[names.length-1];
        	SerializationHelper.write(a + "-" + b + "-" + name + ".model", learner2_3);
        }
        
        
        System.out.println("7...");
        learner3.build(dataset);
        if(saveModels){
        	names = learner3.getClass().toString().split("\\.");
        	a = names[names.length-1];
        	SerializationHelper.write(a + "-" + name + ".model", learner3);
        }
        
        
        System.out.println("8...");
        learner4.build(dataset);
        if(saveModels){
        	names = learner4.getClass().toString().split("\\.");
        	a = names[names.length-1];
        	SerializationHelper.write(a + "-" + name + ".model", learner4);
        }
        
        
        System.out.println("9...");
        learner5.build(dataset);
        if(saveModels){
        	names = learner5.getClass().toString().split("\\.");
        	a = names[names.length-1];
        	SerializationHelper.write(a + "-" + name + ".model", learner5);
        }
        */
        
        System.out.println("Done...");
	}
	
	public void evaluate(String type) throws Exception{
//		Evaluator eval = new Evaluator();
//		MultipleEvaluation results;
		
		
		//  SCORING
		// RAkEL
		// *rakel---k-NN
		int numInstances = test.getNumInstances();
        for ( int instanceIndex = 0; instanceIndex < numInstances; ++instanceIndex ) {
            Instance instance = test.getDataSet().instance(instanceIndex);
            MultiLabelOutput output = learner1_1.makePrediction(instance);
            if ( output.hasBipartition() ) {
//            	output.ranksFromValues(output.getHammingLoss())
                String bipartion = Arrays.toString(output.getBipartition());
                System.out.println("Predicted bipartion: " + bipartion);
            }
        }
		
		/*
		results = eval.crossValidate(learner1_1, dataset, folds);
		result1_1 = results.getMean(type);
		// *rakel---SVM
		results = eval.crossValidate(learner1_2, dataset, folds);
		result1_2 = results.getMean(type);
		// *rakel---J48
		results = eval.crossValidate(learner1_3, dataset, folds);
		result1_3 = results.getMean(type);
		
		// Ensembles of Classifier Chains
		results = eval.crossValidate(learner2_1, dataset, folds);
		result2_1 = results.getMean(type);
		results = eval.crossValidate(learner2_2, dataset, folds);
		result2_2 = results.getMean(type);
		results = eval.crossValidate(learner2_3, dataset, folds);
		result2_3 = results.getMean(type);


		// Evaluation
		//  RANKING
		// Multi-label perceptron
		results = eval.crossValidate(learner3, dataset, folds);
		System.out.println(results);
		result3 = results.getMean(type);

		// Ranking by pairwise comparison
		results = eval.crossValidate(learner4, dataset, folds);
		result4 = results.getMean(type);
		//AdaboostMH
		results = eval.crossValidate(learner5, dataset, folds);
		System.out.println(results);
		result5 = results.getMean(type);*/
		
	}
	
	// Thresholding
	/*private void threshold() {
		//lol
		System.out.println("Does nothing.");
	}*/
	
	public double[] getHammingLoss(){
		double[] res = {result1_1, result1_2, result1_3, result2_1, result2_2, result2_3, result3, result4, result5};
		return res;
	}
	
	@Override
	public String toString(){
		double[] losses = this.getHammingLoss();
		String contents = "\n";
		for(int i=0; i<losses.length; ++i){
    		contents += ("Hamming Loss-classifier" + String.valueOf(i) + ": " + losses[i] + "\n");
    	}
		return contents;
	}
	
}
