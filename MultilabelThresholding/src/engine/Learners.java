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
	private MLkNN learner2;
	
	private int folds;
	
	public Learners( MultiLabelInstances dataset, int folds ) {
		this.dataset = dataset;
		
		this.folds = folds;
		
		this.learner1 = new RAkEL(new LabelPowerset(new J48()));
    	this.learner2 = new MLkNN(); 
	}
	
	public void evaluate(){
		Evaluator eval = new Evaluator();
		MultipleEvaluation results;
		
		results = eval.crossValidate(learner1, dataset, folds);
		System.out.println(results);
		results = eval.crossValidate(learner2, dataset, folds);
		System.out.println(results);
		
		
	}
	
}
