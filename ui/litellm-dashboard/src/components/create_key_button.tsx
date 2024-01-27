'use client';

import React from 'react';
import { Button, TextInput } from '@tremor/react';

import { Card, Metric, Text } from "@tremor/react";

export default function CreateKey() {
  const handleClick = () => {
    console.log('Hello World');
  };

  return (
    // <Card className="max-w-xs mx-auto" decoration="top" decorationColor="indigo">
    //     <Text className='mb-4'>Key Name</Text>
    //     <TextInput className='mb-4' placeholder="My test key"></TextInput>
        
    // </Card>
    <Button className="mx-auto" onClick={handleClick}>+ Create New Key</Button>
  );
}