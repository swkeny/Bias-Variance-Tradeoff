filePathRoot <- 'C:\\Users\\deifen\\Documents\\Projects\\Bias and overfitting trade offs\\Bias and Variance Tradeoff Project\\SampleData\\dataSample'

n <- 750
r <- 50
vect <- seq(n) 
signal<-function(n)
{(n^2.75)/exp(sqrt(n/2)) +.40*n -sqrt(n)}
for(i in 1:10) 
{
  x <- sample(vect, r)
  noise<- rnorm(r, mean=10, sd=20)
  y <- signal(x) + noise
  Data <- data.frame(x, y)
  write.csv(Data, file=paste(filePathRoot, i, collapse=NULL), quote=FALSE, row.names=FALSE, append=FALSE)
}

plot(y ~ x, Data)
curve(signal(x), from=1, to=n, , xlab="x", ylab="y", add=TRUE)

