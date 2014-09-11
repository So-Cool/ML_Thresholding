package engine;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.EuclideanDistance;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.neighboursearch.LinearNNSearch;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;

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
	
	private MLkNN learner2;
	private double result2;
	
	private int folds;
	
	public Learners( MultiLabelInstances dataset, int folds ) throws Exception {
		this.dataset = dataset;
		
		this.folds = folds;
		
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
		
		
    	this.learner2 = new MLkNN(); 
	}
	
	public void evaluate(String type){
		Evaluator eval = new Evaluator();
		MultipleEvaluation results;
		
		// Evaluation
		//  RANKING
		
		results = eval.crossValidate(learner2, dataset, folds);
		result2 = results.getMean(type);
		
		//  SCORING
		// *rakel---k-NN
		results = eval.crossValidate(learner1_1, dataset, folds);
		result1_1 = results.getMean(type);
		// *rakel---SVM
		results = eval.crossValidate(learner1_2, dataset, folds);
		result1_2 = results.getMean(type);
		// *rakel---J48
		results = eval.crossValidate(learner1_3, dataset, folds);
		result1_3 = results.getMean(type);
		
		// *
		// *
		
		// Thresholding
		
	}
	
	public double[] getHammingLoss(){
		double[] res = {result1_1, result1_2, result1_3, result2};
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
