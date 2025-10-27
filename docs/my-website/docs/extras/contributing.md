import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Contributing to Documentation

This website is built using [Docusaurus 2](https://docusaurus.io/), a modern static website generator.

Clone litellm 
```
git clone https://github.com/BerriAI/litellm.git
```

### Local setup for locally running docs

```
cd docs/my-website
```


<Tabs>

<TabItem value="yarn" label="Yarn">

Installation
```
npm install --global yarn
```
Install requirement
```
yarn
```
Run website
```
yarn start
```

</TabItem>

<TabItem value="pnpm" label="pnpm">

Installation
```
npm install --global pnpm
```
Install requirement
```
pnpm install
```
Run website
```
pnpm start
```

</TabItem>

</Tabs>


Open docs here: [http://localhost:3000/](http://localhost:3000/)

This command builds your Markdown files into HTML and starts a development server to browse your documentation. Open up [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your web browser to see your documentation. You can make changes to your Markdown files and your docs will automatically rebuild.

[Full tutorial here](https://docs.readthedocs.io/en/stable/intro/getting-started-with-mkdocs.html)

### Making changes to Docs
- All the docs are placed under the `docs` directory
- If you are adding a new `.md` file or editing the hierarchy edit `mkdocs.yml` in the root of the project
- After testing your changes, make a change/pull request to the `main` branch of [github.com/BerriAI/litellm](https://github.com/BerriAI/litellm)
