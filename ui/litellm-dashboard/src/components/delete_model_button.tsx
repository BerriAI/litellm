"use client";

import React, { useState } from "react";
import { Grid, Col, Icon } from "@tremor/react";
import { Title } from "@tremor/react";
import {
    Modal,
    message,
} from "antd";
import { modelDeleteCall } from "./networking";
import { TrashIcon } from "@heroicons/react/outline";

interface DeleteModelProps {
    modelID: string;
    accessToken: string;
}

const DeleteModelButton: React.FC<DeleteModelProps> = ({
    modelID,
    accessToken,
}) => {
    const [isModalVisible, setIsModalVisible] = useState(false);

    const handleDelete = async () => {
        try {
            message.info("Making API Call");
            setIsModalVisible(true);
            const response = await modelDeleteCall(accessToken, modelID);

            console.log("model delete Response:", response);
            message.success(`Model ${modelID} deleted successfully`);
            setIsModalVisible(false);
        } catch (error) {
            console.error("Error deleting the model:", error);
        }
    };

    return (
        <div>
            <Icon
                onClick={() => setIsModalVisible(true)}
                icon={TrashIcon}
                size="sm"
            />

            <Modal
                open={isModalVisible}
                onOk={handleDelete}
                okType="danger"
                onCancel={() => setIsModalVisible(false)}
            >
                <Grid numItems={1} className="gap-2 w-full">

                    <Title>Delete Model</Title>
                    <Col numColSpan={1}>
                        <p>
                            Are you sure you want to delete this model? This action is irreversible.
                        </p>
                    </Col>
                    <Col numColSpan={1}>
                        <p>
                            Model ID: <b>{modelID}</b>
                        </p>
                    </Col>
                </Grid>
            </Modal>
        </div>
    );
};

export default DeleteModelButton;