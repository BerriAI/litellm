# Contributing to Documentation

Clone litellm 
```
git clone https://github.com/BerriAI/litellm.git
```

### Local setup for locally running docs

#### Installation
```
pip install mkdocs
```

#### Locally Serving Docs
```
mkdocs serve
```
If you see `command not found: mkdocs` try running the following
```
python3 -m mkdocs serve
```

This command builds your Markdown files into HTML and starts a development server to browse your documentation. Open up [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your web browser to see your documentation. You can make changes to your Markdown files and your docs will automatically rebuild.

[Full tutorial here](https://docs.readthedocs.io/en/stable/intro/getting-started-with-mkdocs.html)

### Making changes to Docs
- All the docs are placed under the `docs` directory
- If you are adding a new `.md` file or editing the hierarchy edit `mkdocs.yml` in the root of the project
- After testing your changes, make a change to the `main` branch of [github.com/BerriAI/litellm](https://github.com/BerriAI/litellm)




