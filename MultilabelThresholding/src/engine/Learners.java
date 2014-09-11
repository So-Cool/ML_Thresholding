package engine;

import weka.classifiers.trees.J48;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;

public class Learners {

	private MultiLabelInstances dataset;
	
	private RAkEL learner1;
	private double result1;
	
	private MLkNN learner2;
	private double result2;
	
	private int folds;
	
	public Learners( MultiLabelInstances dataset, int folds ) {
		this.dataset = dataset;
		
		this.folds = folds;
		
		this.learner1 = new RAkEL(new LabelPowerset(new J48()));
    	this.learner2 = new MLkNN(); 
	}
	
	public void evaluate(String type){
		Evaluator eval = new Evaluator();
		MultipleEvaluation results;
		
		// Evaluation
		//  RANKING
		results = eval.crossValidate(learner1, dataset, folds);
		result1 = results.getMean(type);
		
		results = eval.crossValidate(learner2, dataset, folds);
		result2 = results.getMean(type);
		
		//  SCORING
		// *
		// *
		// *
		
		// Thresholding
		
	}
	
	public double[] getHammingLoss(){
		double[] res = {result1, result2};
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
