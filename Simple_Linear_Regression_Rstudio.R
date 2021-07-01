library('lattice')
library(psych)
summary(delivery_time$Delivery.Time)
summary(delivery_time$Sorting.Time)
attach(delivery_time)
dotplot(delivery_time)
boxplot(delivery_time)
qqnorm(Delivery.Time,pch=19)
qqline(Delivery.Time)
qqnorm(Sorting.Time,pch=19)
qqline(Sorting.Time,pch=19)
plot(delivery_time)
plot(Sorting.Time,prob=T)   # this is showing positive relation 
cor(Delivery.Time,Sorting.Time) ## correlation coefficient = 0.8259973 considerable 
hist(Delivery.Time)
hist(Sorting.Time)
plot(Sorting.Time,Delivery.Time,main="Scatter plot", col = "Dodgerblue4",
     col.main="Dodgerblue4",col.lab="green",ylab = "Delivery Time", xlab = "Sorting Time"
     ,pch=19)
reg_sim <- lm(Delivery.Time~Sorting.Time,data = delivery_time)
summary(reg_sim)
confint(reg_sim,level = 0.95)
pred_sim <- predict(reg_sim,interval = 'predict')
pred_sim <- as.data.frame(pred_sim)
cor_val_sim <- cor(pred_sim$fit,delivery_time)  # 0.8259973
cor_val_sim <- as.data.frame(cor_val_sim)
cor_val_sim[1,1]
str(pred_sim)
confint(reg_sim,level = 0.95)

total_result <- data.frame(
  'algoname' =NULL,
  'correlation' = NULL
)

temp.data <- data.frame('reg_sim',cor_val_sim[1,1])
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_sqrt <- lm(Delivery.Time~sqrt(Sorting.Time), data = delivery_time)
summary(reg_sqrt)
confint(reg_sqrt)
pred_sqrt <- predict(reg_sqrt,interval = 'predict')
pred_sqrt
pred_sqrt <- as.data.frame(pred_sqrt)
cor_val_sqrt <- cor(pred_sqrt$fit,Sorting.Time)
cor_val_sqrt
temp.data <- data.frame('reg_sqrt',cor_val_sqrt)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_log <- lm(Sorting.Time~log(Delivery.Time),data = delivery_time)
summary(reg_log)
confint(reg_log)
pred_log <- predict(reg_log,interval='predict')
pred_log
pred_log <- as.data.frame(pred_log)
cor_val_log <- cor(pred_log$fit,Delivery.Time)
cor_val_log
temp.data <- data.frame('reg_log',cor_val_log)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_expo <- lm(log(Delivery.Time)~Sorting.Time + I(Sorting.Time*Sorting.Time),data = delivery_time)
summary(reg_expo)
pred_expo <- predict(reg_expo,interval = 'predict')
pred_expo <- as.data.frame(pred_expo)
pred_expo
pred_expo$fit <- exp(pred_expo$fit)
pred_expo
cor_val_expo <- cor(pred_expo$fit,Delivery.Time)
cor_val_expo
temp.data <- data.frame('pred_expo',cor_val_expo)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_sqrt_dil <- lm(sqrt(Delivery.Time)~Sorting.Time,data = delivery_time)
summary(reg_sqrt_dil)
pred_sqrt_dil <- predict(reg_sqrt_dil,interval = 'predict')
pred_sqrt_dil
pred_sqrt_dil <- as.data.frame(pred_sqrt_dil)
cor_val_sqrt_dil <- cor(pred_sqrt_dil$fit,Delivery.Time)
cor_val_sqrt_dil
temp.data <- data.frame('reg_sqrt_dil',cor_val_sqrt_dil)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_log_at <- lm(log(Delivery.Time)~Sorting.Time,data = delivery_time)
summary(reg_log_at)
pred_log_at <- predict(reg_log_at,interval = 'predict')
pred_log_at
pred_log_at <- as.data.frame(pred_log_at)
pred_log_at$fit <- exp(pred_log_at$fit)
cor_val_log_at <- cor(pred_log_at$fit,Sorting.Time)
cor_val_log_at
temp.data <- data.frame('reg_log_at',cor_val_log_at)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result
