try:
    configs
except NameError:
    configs = {}

try:
    criterion = configs['criterion']
    criterion = eval(f"{criterion}()")
except:
    print("Using default criterion: MSELoss")
    criterion = torch.nn.MSELoss()

try:
    optimizer = configs['optimizer']
    optimizer = eval(f"torch.optim.{optimizer}")
except:
    print("Using default optimizer: Adam")
    optimizer = torch.optim.Adam

lr = configs.get("lr", 5e-4)

num_epochs = configs.get("num_epochs", 150)


kwargs = dict(
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    lr=lr,
    verbose=1,
    mutator=mutator
)