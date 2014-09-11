lib=exlib
classpath=.:bin:$(lib)/weka.jar:$(lib)/mulan.jar
sourcepath=src
memoryoptions=-Xms512m -Xmx512m -server -XX:+UseParNewGC

all: compile_all
run: run_all
test: test
clean: clean_all

compile_all:
	@if [ -z "$${JAVA_HOME}" ] ; then echo "JAVA_HOME not set" & echo "In redHat it might be: /usr/lib/jvm/java-1.7.0-openjdk.x86_64" & echo 'In OS X it is: export JAVA_HOME=$$(/usr/libexec/java_home)' & echo "To set execute: export JAVA_HOME=path/to/your/java/environment" & exit 0 ; fi
	@if [ ! -x "$${JAVA_HOME}/bin/java" ] ; then echo Cannot find java executable, check JAVA_HOME & exit 0 ; fi
	@if [ ! -d "bin" ] ; then mkdir bin ; fi
	@#-source 1.6
	@$${JAVA_HOME}/bin/javac -cp ${classpath} -d bin -sourcepath $(sourcepath) $(sourcepath)/engine/Engine.java

test:
	@$$JAVA_HOME/bin/java ${memoryoptions} -cp ${classpath} engine.Engine datasets/emotions.arff datasets/emotions.xml

run_all:
	@$$JAVA_HOME/bin/java ${memoryoptions} -cp ${classpath} engine.Engine datasets/CAL500/CAL500.arff datasets/CAL500/CAL500.xml
	@$$JAVA_HOME/bin/java ${memoryoptions} -cp ${classpath} engine.Engine datasets/delicious/delicious.arff datasets/delicious/delicious.xml
	@$$JAVA_HOME/bin/java ${memoryoptions} -cp ${classpath} engine.Engine datasets/enron/enron.arff datasets/enron/enron.xml
	@$$JAVA_HOME/bin/java ${memoryoptions} -cp ${classpath} engine.Engine datasets/mediamill/mediamill.arff datasets/mediamill/mediamill.xml
	@$$JAVA_HOME/bin/java ${memoryoptions} -cp ${classpath} engine.Engine datasets/yeast/yeast.arff datasets/yeast/yeast.xml

clean_all:
	@rm -rf bin
