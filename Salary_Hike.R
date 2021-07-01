## Salary Hike 
attach(Salary_Data)
qqnorm(YearsExperience)
qqline(YearsExperience)
hist(YearsExperience)
plot(Salary_Data)  ## Positive correlation 
boxplot(YearsExperience)
boxplot(Salary)
qqnorm(Salary)
hist(Salary)
cor(YearsExperience,Salary)  ## 0.9782416

total_result <- data.frame(
  'algoname' =NULL,
  'correlation' = NULL
)

reg_sim <- lm(Salary~YearsExperience,data = Salary_Data)
summary(reg_sim)
pred_sim <- predict(reg_sim,interval = 'predict')
pred_sim <- as.data.frame(pred_sim)
cor_val_sim <- cor(pred_sim$fit,Salary)
cor_val_sim
temp.data <- data.frame('reg_sim',cor_val_sim)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_sqrt <- lm(Salary~sqrt(YearsExperience),data = Salary_Data)
summary(reg_sqrt)
pred_sqrt <- predict(reg_sqrt,interval = 'predict')
pred_sqrt <- as.data.frame(pred_sqrt)
cor_val_sqrt <- cor(pred_sqrt$fit,Salary)
temp.data <- data.frame('reg_sqrt',cor_val_sqrt)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_log <- lm(Salary~log(YearsExperience),data = Salary_Data)
summary(reg_log)
pred_log <- predict(reg_log,interval = 'predict')
pred_log <- as.data.frame(pred_log)
cor_val_log <- cor(pred_log$fit,Salary)
cor_val_log
temp.data <- data.frame('reg_log',cor_val_log)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_expo <- lm(log(Salary)~YearsExperience+I(YearsExperience*YearsExperience),data = Salary_Data)
summary(reg_expo)
pred_expo <- predict(reg_expo,interval = 'predict')
pred_expo <- as.data.frame(pred_expo)
pred_expo
pred_expo$fit <- exp(pred_expo$fit)
pred_expo
cor_val_expo <- cor(pred_expo$fit,Salary)
cor_val_expo
temp.data <- data.frame('reg_expo',cor_val_expo)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_sqrt_sal <- lm(sqrt(Salary)~YearsExperience,data = Salary_Data)
summary(reg_sqrt_sal)
pred_sqrt_sal <- predict(reg_sqrt_sal,interval = 'predict')
pred_sqrt_sal <- as.data.frame(pred_sqrt_sal)
cor_val_sqrt_sal <- cor(pred_sqrt_sal$fit,Salary)
cor_val_sqrt_sal
temp.data <- data.frame('reg_sqrt_sal',cor_val_sqrt_sal)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result

reg_log_sal <- lm(log(Salary)~YearsExperience,data = Salary_Data)
summary(reg_log_sal)
pred_log_sal <- predict(reg_log_sal,interval = 'predict')
pred_log_sal <- as.data.frame(pred_log_sal)
pred_log_sal
pred_log_sal$fit <- log(pred_log_sal$fit)
cor_val_log_sal <- cor(pred_log_sal$fit,Salary)
cor_val_log_sal
temp.data <- data.frame('reg_log_sal',cor_val_log_sal)
names(temp.data) <- c('algoname','correlation')
total_result <- rbind(total_result,temp.data)
total_result
