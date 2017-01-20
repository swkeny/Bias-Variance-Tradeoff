filePathRoot <- '/Users/wing/Downloads/Bias-Variance-Tradeoff-master/SampleData1/sampledata'

n <- 25
r <- 20
vect <- seq(from= -pi, to =pi, by=.1) 
signal<-function(n)
{sin(n)}
for(i in 1:10) 
{
  x <- sample(vect, r)
  noise<- rnorm(r, mean=0, sd=.1)
  y <- signal(x) + noise
  Data <- data.frame(x, y)
  write.csv(Data, file=paste(filePathRoot, i, collapse=NULL), quote=FALSE, row.names=FALSE)
}

plot(y ~ x, Data)
curve(signal(x), from=-pi, to=pi, , xlab="x", ylab="y", add=TRUE)

