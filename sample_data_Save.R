filePathRootTrain <- 'C:\\Users\\deifen\\Documents\\Projects\\Bias and overfitting trade offs\\Project\\SampleData\\train'
filePathRootTest <- 'C:\\Users\\deifen\\Documents\\Projects\\Bias and overfitting trade offs\\Project\\SampleData\\test'
n <- 25
r <- 20
vect <- seq(from= -pi, to =pi, by=.1) 
signal<-function(n)
{sin(n)}
for(i in 1:10) 
{
  x <- sample(vect, r)
  noise<- rnorm(r, mean=0, sd=.5)
  y <- signal(x) + noise
  trainData <- data.frame(x, y)
  write.csv(trainData, file=paste(filePathRootTrain, i, collapse=NULL), quote=FALSE, row.names=FALSE)
}

x <- sample(vect, r)
noise<- rnorm(r, mean=0, sd=.5)
y <- signal(x) + noise
testData <- data.frame(x, y)
write.csv(testData, file=paste(filePathRootTest, 1, collapse=NULL), quote=FALSE, row.names=FALSE)


plot(y ~ x, trainData)
curve(signal(x), from=-pi, to=pi, , xlab="x", ylab="y", add=TRUE)

plot(y ~ x, testData)
curve(signal(x), from=-pi, to=pi, , xlab="x", ylab="y", add=TRUE)




