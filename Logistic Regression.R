bank <- read.csv("C:/Users/A/Downloads/bank-full (3).csv", sep = ";")
head(bank, 3)
dim(bank)
names(bank)
str(bank)
anyNA(bank)
bank$job <- as.factor(bank$job)
bank$marital <- as.factor(bank$marital)
bank$education <- as.factor(bank$education)
bank$default <- as.factor(bank$default)
bank$housing <- as.factor(bank$housing)
bank$loan <- as.factor(bank$loan)
bank$day <- as.factor(bank$day)
bank$month <- as.factor(bank$month)
bank$poutcome <- as.factor(bank$poutcome)
bank$y <- as.factor(bank$y)

str(bank)
length(unique(bank$day))
length(unique(bank$month))
bank$monthdate <- as.Date(paste(bank$month,bank$day, sep = "-"), format = "%b-%d")
head(bank)
bank$rangeage <- as.factor(case_when(
  bank$age < 23 ~ "18-22",
  bank$age > 22 & bank$age < 36 ~ "23-35",
  bank$age > 35 & bank$age < 51 ~ "36-50",
  TRUE ~ "above 50"
))
head(bank)
sort(round(prop.table(table(bank$job))*100, 2), decreasing = T)
sort(round(prop.table(table(bank$marital))*100, 2), decreasing = T)
sort(round(prop.table(table(bank$education))*100, 2), decreasing = T)
sort(round(prop.table(table(bank$rangeage))*100, 2), decreasing = T)
round(prop.table(table(bank$job, bank$marital))*100, 2)
round(prop.table(table(bank$job, bank$education))*100, 2)
round(prop.table(table(bank$job, bank$loan))*100, 2)
round(prop.table(table(bank$job, bank$housing))*100, 2)
summary(bank$balance)
minus <- bank[bank$balance < 0, ]
aggregate(formula = balance ~ job, data = bank, FUN = mean)
aggregate(formula = balance ~ job, data = minus, FUN = mean)
data <- xtabs( ~ poutcome + job, bank)
data
heatmap(data, Rowv = NA, Colv = NA, cexCol = 1, cexRow = 1, scale = "row")
success <- bank[bank$poutcome == "success",]
success$month <- factor(success$month, levels = c("jan", "feb", "mar","apr","may","jun","jul","aug","sep","oct","nov","dec"))
success_xtabs <- xtabs(~month + job, data = success )
success_xtabs
heatmap(success_xtabs, Rowv = NA, Colv = NA, scale = "row")
subscribe <- xtabs(~ job+y, data = bank)
heatmap(x = subscribe, Rowv = NA, Colv = NA, cexCol = 1, scale = "column")
