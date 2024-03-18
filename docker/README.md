# LiteLLM Docker 

This is a minimal Docker Compose setup for self-hosting LiteLLM.

<div align="center">

 | [日本語](README_JP.md) | [English](README.md) |

</div>


## Usage

1. Clone or download this repository.

2. Navigate to the project directory.

   ```
   cd litellm
   ```

3. Start the containers using Docker Compose.

   ```
   docker-compose up
   ```

   This will download the necessary images and start the LiteLLM container.

4. Open a new terminal and access the running LiteLLM container.

   ```
   docker-compose exec litellm /bin/bash
   ```

   This will give you access to a bash shell inside the LiteLLM container.

5. Perform any desired operations inside the container. For example, you can edit the LiteLLM configuration files or install additional packages.

6. To exit, run the `exit` command inside the container to leave the shell, and press Ctrl+C to stop Docker Compose.

   ```
   exit
   ```

   Then, press Ctrl+C.

## Customization

- You can modify the container configuration by editing the `docker-compose.yml` file as needed.
- The LiteLLM configuration files are located in the `/app/config` directory inside the container. You can edit these files to customize the behavior of LiteLLM.

## Notes

- This setup is provided for development and testing purposes. In a production environment, proper security measures should be implemented.
- Any data generated inside the container will be lost when the container is destroyed. Make sure to backup any necessary data to the host machine.

That's the basic steps to self-host LiteLLM using Docker. If you have any questions, feel free to ask.